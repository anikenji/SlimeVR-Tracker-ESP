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

namespace SlimeVR::MotionProcessing {

/**
 * @brief Bio-Stance & Micro-ZUPT (Zero Velocity Update) Detector for SlimeVR.
 * 
 * In human locomotion, stance phases during foot-ground contact last only 120ms - 250ms,
 * which is too short for standard rest detectors (requiring 2.0 - 3.0 seconds).
 * This module detects sub-second stance phases and performs fast micro-ZUPT bias
 * corrections, preventing runaway yaw drift during active user locomotion.
 */
class BioStanceDetector {
public:
	BioStanceDetector(
		float maxStanceOmegaRad = 0.22f,     // rad/s (~12.6 deg/s) angular velocity ceiling
		float maxStanceAccDev = 0.45f,        // m/s^2 deviation from gravity (~0.045g)
		float minStanceDurationSec = 0.14f,   // 140ms minimum stance window
		float zuptLearningRate = 0.06f        // Exponential filter update gain per stance event
	)
		: m_maxOmega(maxStanceOmegaRad)
		, m_maxAccDev(maxStanceAccDev)
		, m_minStanceDuration(minStanceDurationSec)
		, m_learningRate(zuptLearningRate)
		, m_stanceTime(0.0f)
		, m_isStance(false)
		, m_stanceTriggered(false)
		, m_stanceCount(0)
		, m_sampleAccumCount(0)
	{
		m_biasZupt[0] = 0.0f;
		m_biasZupt[1] = 0.0f;
		m_biasZupt[2] = 0.0f;
		m_accumOmega[0] = 0.0f;
		m_accumOmega[1] = 0.0f;
		m_accumOmega[2] = 0.0f;
	}

	void reset() {
		m_stanceTime = 0.0f;
		m_isStance = false;
		m_stanceTriggered = false;
		m_sampleAccumCount = 0;
		m_accumOmega[0] = 0.0f;
		m_accumOmega[1] = 0.0f;
		m_accumOmega[2] = 0.0f;
	}

	/**
	 * @brief Feed raw gyro (rad/s) and accel (m/s^2) samples.
	 * 
	 * @param gyr Angular velocity in rad/s [x, y, z]
	 * @param acc Acceleration in m/s^2 [x, y, z]
	 * @param dt Timestep in seconds
	 * @return true if a new micro-ZUPT stance event was just triggered
	 */
	bool update(const float gyr[3], const float acc[3], float dt) {
		m_stanceTriggered = false;

		float omegaNorm = std::sqrt(gyr[0] * gyr[0] + gyr[1] * gyr[1] + gyr[2] * gyr[2]);
		float accNorm = std::sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
		float accDev = std::fabs(accNorm - 9.80665f);

		bool conditionMet = (omegaNorm < m_maxOmega) && (accDev < m_maxAccDev);

		if (conditionMet) {
			m_stanceTime += dt;
			m_accumOmega[0] += gyr[0];
			m_accumOmega[1] += gyr[1];
			m_accumOmega[2] += gyr[2];
			m_sampleAccumCount++;

			if (!m_isStance && m_stanceTime >= m_minStanceDuration) {
				m_isStance = true;
				m_stanceTriggered = true;
				m_stanceCount++;
				applyZuptUpdate();
			}
		} else {
			if (m_isStance && m_sampleAccumCount > 0) {
				applyZuptUpdate();
			}
			m_stanceTime = 0.0f;
			m_isStance = false;
			m_sampleAccumCount = 0;
			m_accumOmega[0] = 0.0f;
			m_accumOmega[1] = 0.0f;
			m_accumOmega[2] = 0.0f;
		}

		return m_stanceTriggered;
	}

	/**
	 * @brief Correct raw gyro sample by subtracting the Micro-ZUPT estimated bias.
	 */
	void correctGyro(const float inGyr[3], float outGyr[3]) const {
		outGyr[0] = inGyr[0] - m_biasZupt[0];
		outGyr[1] = inGyr[1] - m_biasZupt[1];
		outGyr[2] = inGyr[2] - m_biasZupt[2];
	}

	[[nodiscard]] bool isStance() const { return m_isStance; }
	[[nodiscard]] bool wasTriggered() const { return m_stanceTriggered; }
	[[nodiscard]] uint32_t getStanceCount() const { return m_stanceCount; }
	[[nodiscard]] const float* getZuptBias() const { return m_biasZupt; }

	void setZuptBias(float bx, float by, float bz) {
		m_biasZupt[0] = bx;
		m_biasZupt[1] = by;
		m_biasZupt[2] = bz;
	}

private:
	float m_maxOmega;
	float m_maxAccDev;
	float m_minStanceDuration;
	float m_learningRate;

	float m_stanceTime;
	bool m_isStance;
	bool m_stanceTriggered;
	uint32_t m_stanceCount;

	uint32_t m_sampleAccumCount;
	float m_accumOmega[3];
	float m_biasZupt[3];

	void applyZuptUpdate() {
		if (m_sampleAccumCount == 0) return;

		float meanGyr[3];
		meanGyr[0] = m_accumOmega[0] / static_cast<float>(m_sampleAccumCount);
		meanGyr[1] = m_accumOmega[1] / static_cast<float>(m_sampleAccumCount);
		meanGyr[2] = m_accumOmega[2] / static_cast<float>(m_sampleAccumCount);

		// Limit step size to avoid jump in case of momentary shock
		constexpr float maxStep = 0.05f; // rad/s
		for (int i = 0; i < 3; ++i) {
			float delta = m_learningRate * (meanGyr[i] - m_biasZupt[i]);
			delta = std::clamp(delta, -maxStep, maxStep);
			m_biasZupt[i] += delta;
		}

		m_sampleAccumCount = 0;
		m_accumOmega[0] = 0.0f;
		m_accumOmega[1] = 0.0f;
		m_accumOmega[2] = 0.0f;
	}
};

} // namespace SlimeVR::MotionProcessing
