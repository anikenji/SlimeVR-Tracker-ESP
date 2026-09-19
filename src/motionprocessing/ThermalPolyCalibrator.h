/*
	SlimeVR Code is placed under the MIT license
	Copyright (c) 2025 SlimeVR Contributors

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "OnlinePolyfit.h"
#include "types.h"

namespace SlimeVR::MotionProcessing {

/**
 * @brief Real-time thermal polynomial compensator for IMU gyro Zero-Rate Offset (ZRO) drift.
 *
 * Models gyro zero-rate offset drift as a polynomial function of temperature:
 *   b_i(T) = c0_i + c1_i * (T - T0) + c2_i * (T - T0)^2 [+ c3_i * (T - T0)^3]
 * where T0 is a reference temperature (default 23.0 deg C).
 *
 * Employs Recursive Least Squares (RLS) with Givens rotations via OnlineVectorPolyfit
 * to continuously learn or update the thermal drift curve during periods when the
 * tracker is at rest.
 */
template <uint32_t Degree = 2, uint64_t ForgettingSamples = 1000000ULL>
class ThermalPolyCalibratorTemplate {
public:
	static constexpr uint32_t PolyDegree = Degree;
	static constexpr uint32_t NumDimensions = 3;
	static constexpr uint32_t NumCoefficients = Degree + 1;

	static constexpr float DefaultReferenceTemp = 23.0f;
	static constexpr float MinValidTemp = 10.0f;
	static constexpr float MaxValidTemp = 60.0f;
	static constexpr float MinTempSpanForPolyFit = 0.5f;  // deg C span required for > 0th order fit
	static constexpr uint32_t MinSamplesForFit = 3;

	ThermalPolyCalibratorTemplate(float referenceTemp = DefaultReferenceTemp)
		: m_referenceTemp(referenceTemp) {
		reset();
	}

	/**
	 * @brief Reset the calibrator to uncalibrated state.
	 */
	void reset() {
		m_polyfit.reset();
		std::fill(&m_coeffs[0][0], &m_coeffs[0][0] + NumDimensions * NumCoefficients, 0.0f);
		m_isCalibrated = false;
		m_samplesFed = 0;
		m_minTempObserved = 1000.0f;
		m_maxTempObserved = -1000.0f;
	}

	/**
	 * @brief Whether valid thermal calibration coefficients are currently loaded or trained.
	 */
	[[nodiscard]] bool isCalibrated() const {
		return m_isCalibrated;
	}

	/**
	 * @brief Get the reference temperature T0 (deg C).
	 */
	[[nodiscard]] float getReferenceTemperature() const {
		return m_referenceTemp;
	}

	/**
	 * @brief Set the reference temperature T0 (deg C).
	 */
	void setReferenceTemperature(float referenceTemp) {
		if (std::isfinite(referenceTemp)) {
			m_referenceTemp = referenceTemp;
		}
	}

	/**
	 * @brief Predict the gyro bias for a given axis at a specific temperature.
	 * 
	 * Evaluates b_i(T) using Horner's method:
	 *   b_i(T) = c0_i + dT * (c1_i + dT * (c2_i + ...))
	 * Clamps temperature to [MinValidTemp, MaxValidTemp] to guard against extrapolation runaway.
	 * Returns 0.0f if uncalibrated or on invalid input.
	 */
	[[nodiscard]] float predictBias(int axis, float temperature) const {
		if (!m_isCalibrated || axis < 0 || axis >= static_cast<int>(NumDimensions)) {
			return 0.0f;
		}
		if (!std::isfinite(temperature)) {
			return 0.0f;
		}

		const float clampedTemp = std::clamp(temperature, MinValidTemp, MaxValidTemp);
		const float dT = clampedTemp - m_referenceTemp;

		float bias = m_coeffs[axis][PolyDegree];
		for (int i = static_cast<int>(PolyDegree) - 1; i >= 0; i--) {
			bias = bias * dT + m_coeffs[axis][i];
		}

		return std::isfinite(bias) ? bias : 0.0f;
	}

	/**
	 * @brief Apply thermal polynomial correction by subtracting the estimated bias from gyro[3].
	 *
	 * b_i(T) = c0_i + c1_i * (T - T0) + c2_i * (T - T0)^2
	 * gyro[i] -= b_i(T)
	 *
	 * Robust against uncalibrated state, NaN/Inf, and out-of-range temperatures.
	 */
	void applyCorrection(float temperature, sensor_real_t gyro[3]) const {
		if (!m_isCalibrated || gyro == nullptr) {
			return;
		}
		if (!std::isfinite(temperature)) {
			return;
		}

		const float clampedTemp = std::clamp(temperature, MinValidTemp, MaxValidTemp);
		const float dT = clampedTemp - m_referenceTemp;

		for (uint32_t d = 0; d < NumDimensions; d++) {
			if (!std::isfinite(gyro[d])) {
				continue;
			}

			// Horner's method
			float bias = m_coeffs[d][PolyDegree];
			for (int i = static_cast<int>(PolyDegree) - 1; i >= 0; i--) {
				bias = bias * dT + m_coeffs[d][i];
			}

			if (std::isfinite(bias)) {
				gyro[d] -= static_cast<sensor_real_t>(bias);
			}
		}
	}

	/**
	 * @brief Feed a sample to OnlineVectorPolyfit during rest periods to train/update coefficients.
	 *
	 * Only updates when isResting is true and inputs are finite numbers.
	 * Temperature is clamped to [MinValidTemp, MaxValidTemp].
	 */
	void feedSample(float temperature, const sensor_real_t gyro[3], bool isResting) {
		if (!isResting || gyro == nullptr) {
			return;
		}
		if (!std::isfinite(temperature)) {
			return;
		}
		for (uint32_t d = 0; d < NumDimensions; d++) {
			if (!std::isfinite(gyro[d])) {
				return;
			}
		}

		const float clampedTemp = std::clamp(temperature, MinValidTemp, MaxValidTemp);
		const float dT = clampedTemp - m_referenceTemp;

		const double yValues[NumDimensions] = {
			static_cast<double>(gyro[0]),
			static_cast<double>(gyro[1]),
			static_cast<double>(gyro[2])
		};

		m_polyfit.update(static_cast<double>(dT), yValues);
		m_samplesFed++;

		if (clampedTemp < m_minTempObserved) {
			m_minTempObserved = clampedTemp;
		}
		if (clampedTemp > m_maxTempObserved) {
			m_maxTempObserved = clampedTemp;
		}

		if (m_samplesFed >= MinSamplesForFit) {
			updateCoefficients();
		}
	}

	/**
	 * @brief Recompute polynomial coefficients from RLS Givens matrix.
	 */
	void updateCoefficients() {
		if (m_samplesFed < MinSamplesForFit) {
			return;
		}

		const auto computed = m_polyfit.computeCoefficients();
		const float tempSpan = m_maxTempObserved - m_minTempObserved;

		bool valid = true;
		for (uint32_t d = 0; d < NumDimensions; d++) {
			for (uint32_t c = 0; c < NumCoefficients; c++) {
				if (!std::isfinite(computed[d][c])) {
					valid = false;
					break;
				}
			}
		}

		if (!valid) {
			return;
		}

		if (tempSpan < MinTempSpanForPolyFit) {
			// Temperature span is too narrow to stably resolve 1st/2nd order terms.
			// Fit only constant offset c0, set higher orders to 0.
			for (uint32_t d = 0; d < NumDimensions; d++) {
				m_coeffs[d][0] = computed[d][0];
				for (uint32_t c = 1; c < NumCoefficients; c++) {
					m_coeffs[d][c] = 0.0f;
				}
			}
		} else {
			for (uint32_t d = 0; d < NumDimensions; d++) {
				for (uint32_t c = 0; c < NumCoefficients; c++) {
					m_coeffs[d][c] = computed[d][c];
				}
			}
		}

		m_isCalibrated = true;
	}

	/**
	 * @brief Retrieve 3x3 coefficients: coeffs[axis][c0, c1, c2].
	 */
	void getCoefficients(float coeffs[3][3]) const {
		if (coeffs == nullptr) {
			return;
		}
		for (uint32_t d = 0; d < 3; d++) {
			for (uint32_t c = 0; c < 3; c++) {
				coeffs[d][c] = (c < NumCoefficients) ? m_coeffs[d][c] : 0.0f;
			}
		}
	}

	/**
	 * @brief Set 3x3 coefficients from configuration: coeffs[axis][c0, c1, c2].
	 * Validates against NaN/Inf; marks calibrated if non-zero.
	 */
	void setCoefficients(const float coeffs[3][3]) {
		if (coeffs == nullptr) {
			return;
		}

		bool allFinite = true;
		bool hasNonZero = false;

		for (uint32_t d = 0; d < 3; d++) {
			for (uint32_t c = 0; c < 3; c++) {
				if (!std::isfinite(coeffs[d][c])) {
					allFinite = false;
					break;
				}
				if (coeffs[d][c] != 0.0f) {
					hasNonZero = true;
				}
			}
		}

		if (!allFinite) {
			return;
		}

		for (uint32_t d = 0; d < 3; d++) {
			for (uint32_t c = 0; c < NumCoefficients; c++) {
				m_coeffs[d][c] = (c < 3) ? coeffs[d][c] : 0.0f;
			}
		}

		m_isCalibrated = hasNonZero;

		if (hasNonZero) {
			seedPolyfitFromCoeffs();
		}
	}

	[[nodiscard]] uint32_t getSampleCount() const {
		return m_samplesFed;
	}

	[[nodiscard]] float getMinObservedTemp() const {
		return m_minTempObserved;
	}

	[[nodiscard]] float getMaxObservedTemp() const {
		return m_maxTempObserved;
	}

private:
	/**
	 * @brief Seed RLS matrix with loaded coefficients so online training starts from saved curve.
	 */
	void seedPolyfitFromCoeffs() {
		m_polyfit.reset();
		const float seedPoints[] = {15.0f, 20.0f, 23.0f, 30.0f, 40.0f};
		for (float t : seedPoints) {
			const float dT = t - m_referenceTemp;
			double y[NumDimensions];
			for (uint32_t d = 0; d < NumDimensions; d++) {
				float val = m_coeffs[d][PolyDegree];
				for (int i = static_cast<int>(PolyDegree) - 1; i >= 0; i--) {
					val = val * dT + m_coeffs[d][i];
				}
				y[d] = static_cast<double>(val);
			}
			for (int rep = 0; rep < 10; rep++) {
				m_polyfit.update(static_cast<double>(dT), y);
			}
		}
		m_samplesFed = 50;
		m_minTempObserved = 15.0f;
		m_maxTempObserved = 40.0f;
	}

	float m_referenceTemp = DefaultReferenceTemp;
	float m_coeffs[NumDimensions][NumCoefficients] = {{0.0f}};
	bool m_isCalibrated = false;

	uint32_t m_samplesFed = 0;
	float m_minTempObserved = 1000.0f;
	float m_maxTempObserved = -1000.0f;

	OnlineVectorPolyfit<Degree, NumDimensions, ForgettingSamples> m_polyfit;
};

/// Standard 2nd-order thermal polynomial calibrator (Degree = 2: c0, c1, c2 per axis)
using ThermalPolyCalibrator = ThermalPolyCalibratorTemplate<2>;

}  // namespace SlimeVR::MotionProcessing

namespace SlimeVR {
using MotionProcessing::ThermalPolyCalibrator;
using MotionProcessing::ThermalPolyCalibratorTemplate;
}  // namespace SlimeVR
