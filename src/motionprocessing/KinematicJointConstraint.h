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

#include <cmath>
#include <cstdint>
#include <algorithm>
#include "quat.h"
#include "vector3.h"

namespace SlimeVR::MotionProcessing {

/**
 * @brief Seel-Schauer Kinematic Joint Constraint for Multi-Tracker Systems.
 * 
 * Exploits the biomechanical hinge-joint constraint (e.g. Knee joint between Thigh and Shank)
 * to automatically identify the shared joint axis (j1 in sensor 1, j2 in sensor 2) from angular
 * velocities during flexion/extension, and dynamically eliminates relative yaw drift between
 * the two trackers without requiring magnetometers.
 *
 * Reference:
 * T. Seel, J. Raisch, T. Schauer, "IMU-Based Joint Angle Measurement for Gait Analysis",
 * Sensors 2014, 14(4), 6891-6909.
 */
class KinematicJointConstraint {
public:
	KinematicJointConstraint(
		float minAngularVelocity = 0.35f,   // rad/s threshold for flexion motion
		uint16_t requiredSamples = 180,      // Number of motion samples to identify axes
		float correctionRate = 0.05f,        // Proportional correction gain (learning rate)
		float deadbandAngleRad = 0.035f      // ~2.0 degrees deadband to avoid over-correcting
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

	/**
	 * @brief Feed raw angular velocity samples from tracker 1 (e.g. Thigh) and tracker 2 (e.g. Shank).
	 * Accumulates correlation matrices during active flexion/extension.
	 */
	void feedAngularVelocities(const Vector3& omega1, const Vector3& omega2) {
		float norm1 = omega1.length();
		float norm2 = omega2.length();

		// Only learn from genuine joint rotation (flexion/extension above threshold)
		if (norm1 < m_minOmega || norm2 < m_minOmega) {
			return;
		}

		// Outer products accumulated for Principal Component Analysis (PCA)
		accumulateMatrix(m_C1, omega1);
		accumulateMatrix(m_C2, omega2);
		m_sampleCount++;

		if (!m_identified && m_sampleCount >= m_requiredSamples) {
			computeJointAxes();
		}
	}

	/**
	 * @brief Calculate relative yaw drift correction quaternion for Tracker 2 with respect to Tracker 1.
	 * 
	 * @param q1 World rotation quaternion of Tracker 1 (Thigh)
	 * @param q2 World rotation quaternion of Tracker 2 (Shank)
	 * @return Quat Correction quaternion to be applied to q2 (identity if not ready or in deadband)
	 */
	Quat calculateCorrection(const Quat& q1, const Quat& q2) const {
		if (!m_identified) {
			return Quat(); // Identity quaternion
		}

		// Transform joint axes into world frame
		Vector3 v1 = q1.xform(m_j1);
		Vector3 v2 = q2.xform(m_j2);

		// Project onto horizontal plane (XY) to inspect yaw deviation
		float h1_len = std::sqrt(v1.x * v1.x + v1.y * v1.y);
		float h2_len = std::sqrt(v2.x * v2.x + v2.y * v2.y);

		// If hinge axis is nearly vertical, horizontal projection is ill-conditioned
		if (h1_len < 0.15f || h2_len < 0.15f) {
			return Quat();
		}

		float h1x = v1.x / h1_len;
		float h1y = v1.y / h1_len;
		float h2x = v2.x / h2_len;
		float h2y = v2.y / h2_len;

		// 2D cross product and dot product gives relative yaw drift
		float sinYaw = h1x * h2y - h1y * h2x;
		float cosYaw = h1x * h2x + h1y * h2y;
		float deltaYaw = std::atan2(sinYaw, cosYaw);

		// Check deadband to prevent jitter
		if (std::fabs(deltaYaw) < m_deadband) {
			return Quat();
		}

		// Reject extreme outliers (> 50 deg) which may indicate non-hinge motion (e.g. crossed legs)
		if (std::fabs(deltaYaw) > 0.87f) {
			return Quat();
		}

		// Return corrective rotation about world Z axis
		float correctionAngle = -m_correctionRate * deltaYaw;
		return Quat(Vector3(0.0f, 0.0f, 1.0f), correctionAngle);
	}

	[[nodiscard]] bool isIdentified() const { return m_identified; }
	[[nodiscard]] Vector3 getJointAxis1() const { return m_j1; }
	[[nodiscard]] Vector3 getJointAxis2() const { return m_j2; }
	[[nodiscard]] uint16_t getSampleCount() const { return m_sampleCount; }

	void setJointAxes(const Vector3& j1, const Vector3& j2) {
		m_j1 = j1.normalized();
		m_j2 = j2.normalized();
		m_identified = true;
	}

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
		C[0][0] += v.x * v.x;
		C[0][1] += v.x * v.y;
		C[0][2] += v.x * v.z;
		C[1][0] += v.y * v.x;
		C[1][1] += v.y * v.y;
		C[1][2] += v.y * v.z;
		C[2][0] += v.z * v.x;
		C[2][1] += v.z * v.y;
		C[2][2] += v.z * v.z;
	}

	/**
	 * @brief Power iteration to find dominant eigenvector of a symmetric 3x3 covariance matrix.
	 */
	static Vector3 computeDominantEigenvector(const float C[3][3], int maxIterations = 25) {
		Vector3 v(1.0f, 1.0f, 1.0f);
		v.normalize();

		for (int iter = 0; iter < maxIterations; ++iter) {
			Vector3 next(
				C[0][0] * v.x + C[0][1] * v.y + C[0][2] * v.z,
				C[1][0] * v.x + C[1][1] * v.y + C[1][2] * v.z,
				C[2][0] * v.x + C[2][1] * v.y + C[2][2] * v.z
			);
			float len = next.length();
			if (len < 1e-6f) {
				break;
			}
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
