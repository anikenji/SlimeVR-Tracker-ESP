/*
	SlimeVR Code is placed under the MIT license
	Copyright (c) 2025 SlimeVR Contributors
*/

#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include "quat.h"
#include "vector3.h"

namespace SlimeVR::MotionProcessing {

/**
 * @brief Seel-Schauer Kinematic Joint Constraint for Multi-Tracker Systems.
 * 
 * Automatically identifies shared hinge joint axis (e.g. Knee joint between Thigh and Shank)
 * and eliminates relative yaw drift without requiring magnetometers.
 */
class KinematicJointConstraint {
public:
	KinematicJointConstraint(
		float minAngularVelocity = 0.35f,   // rad/s threshold for flexion motion
		uint16_t requiredSamples = 180,      // Samples to identify axes
		float correctionRate = 0.03f,        // Soft proportional correction gain
		float deadbandAngleRad = 0.045f      // ~2.5 degrees deadband to avoid jitter
	)
		: m_minOmega(minAngularVelocity)
		, m_requiredSamples(requiredSamples)
		, m_correctionRate(correctionRate)
		, m_deadband(deadbandAngleRad)
		, m_identified(false)
		, m_sampleCount(0)
		, m_j1(1.0f, 0.0f, 0.0f)
		, m_j2(1.0f, 0.0f, 0.0f)
	{
		resetMatrices();
	}

	void reset() {
		m_identified = false;
		m_sampleCount = 0;
		resetMatrices();
	}

	void feedAngularVelocities(const Vector3& omega1, const Vector3& omega2) {
		float norm1 = omega1.length();
		float norm2 = omega2.length();

		if (norm1 < m_minOmega || norm2 < m_minOmega) return;

		accumulateMatrix(m_C1, omega1);
		accumulateMatrix(m_C2, omega2);
		m_sampleCount++;

		if (!m_identified && m_sampleCount >= m_requiredSamples) {
			computeJointAxes();
		}
	}

	Quat calculateCorrection(const Quat& q1, const Quat& q2) const {
		if (!m_identified) return Quat();

		Vector3 v1 = q1.xform(m_j1);
		Vector3 v2 = q2.xform(m_j2);

		float h1_len = std::sqrt(v1.x * v1.x + v1.y * v1.y);
		float h2_len = std::sqrt(v2.x * v2.x + v2.y * v2.y);

		if (h1_len < 0.15f || h2_len < 0.15f) return Quat();

		float h1x = v1.x / h1_len;
		float h1y = v1.y / h1_len;
		float h2x = v2.x / h2_len;
		float h2y = v2.y / h2_len;

		float sinYaw = h1x * h2y - h1y * h2x;
		float cosYaw = h1x * h2x + h1y * h2y;
		float deltaYaw = std::atan2(sinYaw, cosYaw);

		if (std::fabs(deltaYaw) < m_deadband) return Quat();
		if (std::fabs(deltaYaw) > 0.80f) return Quat();

		float correctionAngle = -m_correctionRate * deltaYaw;
		return Quat(Vector3(0.0f, 0.0f, 1.0f), correctionAngle);
	}

	[[nodiscard]] bool isIdentified() const { return m_identified; }

private:
	float m_minOmega;
	uint16_t m_requiredSamples;
	float m_correctionRate;
	float m_deadband;
	bool m_identified;
	uint16_t m_sampleCount;

	Vector3 m_j1;
	Vector3 m_j2;

	float m_C1[3][3];
	float m_C2[3][3];

	void resetMatrices() {
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				m_C1[r][c] = 0.0f;
				m_C2[r][c] = 0.0f;
			}
		}
	}

	static void accumulateMatrix(float C[3][3], const Vector3& v) {
		C[0][0] += v.x * v.x; C[0][1] += v.x * v.y; C[0][2] += v.x * v.z;
		C[1][0] += v.y * v.x; C[1][1] += v.y * v.y; C[1][2] += v.y * v.z;
		C[2][0] += v.z * v.x; C[2][1] += v.z * v.y; C[2][2] += v.z * v.z;
	}

	static Vector3 computeDominantEigenvector(const float C[3][3], int maxIterations = 20) {
		Vector3 v(1.0f, 1.0f, 1.0f);
		v.normalize();
		for (int iter = 0; iter < maxIterations; ++iter) {
			Vector3 next(
				C[0][0] * v.x + C[0][1] * v.y + C[0][2] * v.z,
				C[1][0] * v.x + C[1][1] * v.y + C[1][2] * v.z,
				C[2][0] * v.x + C[2][1] * v.y + C[2][2] * v.z
			);
			float len = next.length();
			if (len < 1e-6f) break;
			v = next * (1.0f / len);
		}
		return v;
	}

	void computeJointAxes() {
		m_j1 = computeDominantEigenvector(m_C1);
		m_j2 = computeDominantEigenvector(m_C2);
		if (m_j1.length() > 0.9f && m_j2.length() > 0.9f) {
			m_j1.normalize();
			m_j2.normalize();
			m_identified = true;
		}
	}
};

} // namespace SlimeVR::MotionProcessing
