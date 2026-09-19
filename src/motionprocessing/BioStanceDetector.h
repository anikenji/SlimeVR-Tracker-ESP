/*
	SlimeVR Code is placed under the MIT license
	Copyright (c) 2025 SlimeVR Contributors
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace SlimeVR::MotionProcessing {

/**
 * @brief Bio-Stance & Micro-ZUPT (Zero Velocity Update) Detector for SlimeVR.
 * 
 * Exploits the natural human gait cycle stance phase (120ms - 250ms of ground contact)
 * to gently pull gyro bias towards zero, stopping long-term yaw drift during active VR gameplay.
 * Features strict clamping and sanity bounds to guarantee filter stability.
 */
class BioStanceDetector {
public:
	BioStanceDetector(
		float maxStanceOmegaRad = 0.16f,      // rad/s (~9 deg/s) angular velocity ceiling
		float maxStanceAccDev = 0.40f,        // m/s^2 deviation from gravity (~0.04g)
		float minStanceDurationSec = 0.14f,   // 140ms minimum stance window
		float zuptLearningRate = 0.04f        // Soft exponential filter gain per stance event
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

	bool update(const float gyr[3], const float acc[3], float dt) {
		m_stanceTriggered = false;

		if (dt <= 0.0f || dt > 0.1f) return false;

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

	void correctGyro(const float inGyr[3], float outGyr[3]) const {
		outGyr[0] = inGyr[0] - m_biasZupt[0];
		outGyr[1] = inGyr[1] - m_biasZupt[1];
		outGyr[2] = inGyr[2] - m_biasZupt[2];
	}

	[[nodiscard]] bool isStance() const { return m_isStance; }
	[[nodiscard]] bool wasTriggered() const { return m_stanceTriggered; }
	[[nodiscard]] uint32_t getStanceCount() const { return m_stanceCount; }
	[[nodiscard]] const float* getZuptBias() const { return m_biasZupt; }

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

		// Fail-safe: Step size bounded to 0.01 rad/s (~0.57 deg/s) per stance phase
		constexpr float maxStep = 0.01f;
		// Hard ceiling on total ZUPT bias (0.08 rad/s ~ 4.5 deg/s) to prevent runaway
		constexpr float maxTotalZuptBias = 0.08f;

		for (int i = 0; i < 3; ++i) {
			float delta = m_learningRate * (meanGyr[i] - m_biasZupt[i]);
			delta = std::clamp(delta, -maxStep, maxStep);
			m_biasZupt[i] = std::clamp(m_biasZupt[i] + delta, -maxTotalZuptBias, maxTotalZuptBias);
		}

		m_sampleAccumCount = 0;
		m_accumOmega[0] = 0.0f;
		m_accumOmega[1] = 0.0f;
		m_accumOmega[2] = 0.0f;
	}
};

} // namespace SlimeVR::MotionProcessing
