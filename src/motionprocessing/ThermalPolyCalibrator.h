/*
	SlimeVR Code is placed under the MIT license
	Copyright (c) 2025 SlimeVR Contributors
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "OnlinePolyfit.h"
#include "types.h"

namespace SlimeVR::MotionProcessing {

/**
 * @brief Gated 2nd-order polynomial thermal drift compensator for MEMS IMUs (especially BMI160).
 * 
 * Fits and cancels Zero-Rate Offset (ZRO) drift as a function of sensor temperature:
 *   b_i(T) = c0_i + c1_i * (T - T0) + c2_i * (T - T0)^2
 * 
 * Fail-safe architecture:
 * Remains inactive until at least 150 resting samples across at least a 3.5°C temperature span
 * have been observed. Hard-clamps max correction to prevent wild rotation.
 */
class ThermalPolyCalibrator {
public:
	static constexpr float ReferenceTemperature = 23.0f; // T0 in degrees C
	static constexpr float MinValidTemperature = 10.0f;   // Clamp bound
	static constexpr float MaxValidTemperature = 65.0f;   // Clamp bound
	static constexpr uint32_t MinSamplesRequired = 150;
	static constexpr float MinTempSpanRequired = 3.5f;

	ThermalPolyCalibrator()
		: m_isCalibrated(false)
		, m_samplesFed(0)
		, m_minTempObserved(1000.0f)
		, m_maxTempObserved(-1000.0f)
	{
		for (int d = 0; d < 3; ++d) {
			for (int c = 0; c < 3; ++c) {
				m_coeffs[d][c] = 0.0f;
			}
		}
	}

	void reset() {
		m_polyfit.reset();
		m_isCalibrated = false;
		m_samplesFed = 0;
		m_minTempObserved = 1000.0f;
		m_maxTempObserved = -1000.0f;
		for (int d = 0; d < 3; ++d) {
			for (int c = 0; c < 3; ++c) {
				m_coeffs[d][c] = 0.0f;
			}
		}
	}

	void feedSample(float temperature, const sensor_real_t gyro[3], bool isResting) {
		if (!isResting || gyro == nullptr) {
			return;
		}

		if (!std::isfinite(temperature) || temperature < MinValidTemperature || temperature > MaxValidTemperature) {
			return;
		}

		for (int d = 0; d < 3; ++d) {
			if (!std::isfinite(gyro[d])) return;
		}

		double deltaT = static_cast<double>(temperature - ReferenceTemperature);
		double yValues[3] = {
			static_cast<double>(gyro[0]),
			static_cast<double>(gyro[1]),
			static_cast<double>(gyro[2])
		};

		m_polyfit.update(deltaT, yValues);
		m_samplesFed++;

		if (temperature < m_minTempObserved) m_minTempObserved = temperature;
		if (temperature > m_maxTempObserved) m_maxTempObserved = temperature;

		// Only calculate polynomial after observing at least 3.5°C span and 150 rest samples
		float observedSpan = m_maxTempObserved - m_minTempObserved;
		if (m_samplesFed >= MinSamplesRequired && observedSpan >= MinTempSpanRequired) {
			const auto& computed = m_polyfit.computeCoefficients();
			bool allFinite = true;
			for (int d = 0; d < 3; ++d) {
				for (int c = 0; c < 3; ++c) {
					if (!std::isfinite(computed[d][c])) {
						allFinite = false;
						break;
					}
				}
			}

			if (allFinite) {
				for (int d = 0; d < 3; ++d) {
					for (int c = 0; c < 3; ++c) {
						m_coeffs[d][c] = computed[d][c];
					}
				}
				m_isCalibrated = true;
			}
		}
	}

	void applyCorrection(float temperature, sensor_real_t gyro[3]) const {
		// Fail-safe: If not calibrated with sufficient temperature span, do NOTHING
		if (!m_isCalibrated || gyro == nullptr) {
			return;
		}

		if (!std::isfinite(temperature)) return;

		float clampedT = std::clamp(temperature, MinValidTemperature, MaxValidTemperature);
		float deltaT = clampedT - ReferenceTemperature;

		// Horner evaluation: b = c0 + deltaT * (c1 + deltaT * c2)
		constexpr float maxThermalBiasCap = 0.12f; // ~6.8 deg/s maximum thermal adjustment

		for (int d = 0; d < 3; ++d) {
			float bias = m_coeffs[d][0] + deltaT * (m_coeffs[d][1] + deltaT * m_coeffs[d][2]);
			bias = std::clamp(bias, -maxThermalBiasCap, maxThermalBiasCap);
			gyro[d] -= static_cast<sensor_real_t>(bias);
		}
	}

	[[nodiscard]] bool isCalibrated() const { return m_isCalibrated; }

private:
	OnlineVectorPolyfit<2, 3, 3000> m_polyfit;
	bool m_isCalibrated;
	uint32_t m_samplesFed;
	float m_minTempObserved;
	float m_maxTempObserved;
	float m_coeffs[3][3];
};

} // namespace SlimeVR::MotionProcessing
