#pragma once

#include <vector>

#include <Eigen/Dense>

#include "cloud.h"

struct RigidTransform
{
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

// Running totals over matched pairs (source point, target point). Everything
// the alignment needs fits in these 15 numbers plus a count, so they can be
// accumulated during matching, one copy per thread, and added together after.
// Sums are in double: adding thousands of floats into one running total loses precision.
// Defined in the header so add() inlines into the hot matching loop.
struct PairSums
{
	Eigen::Vector3d sourceSum = Eigen::Vector3d::Zero(); // sum of source points
	Eigen::Vector3d targetSum = Eigen::Vector3d::Zero(); // sum of matched target points
	Eigen::Matrix3d outerSum = Eigen::Matrix3d::Zero();  // sum of (source point) * (target point)^T
	long count = 0;

	void add(const Point& source, const Point& target)
	{
		const Eigen::Vector3d s(source.x, source.y, source.z);
		const Eigen::Vector3d d(target.x, target.y, target.z);
		sourceSum += s;
		targetSum += d;
		outerSum += s * d.transpose();
		++count;
	}

	PairSums& operator+=(const PairSums& other)
	{
		sourceSum += other.sourceSum;
		targetSum += other.targetSum;
		outerSum += other.outerSum;
		count += other.count;
		return *this;
	}
};

// Lets PairSums be used in `reduction(+ : ...)`: each thread gets its own
// default-constructed (zero) copy, and OpenMP adds them together at the end.
#pragma omp declare reduction(+ : PairSums : omp_out += omp_in)

// Best-fit R and t from accumulated pair sums (least squares, Kabsch/SVD method).
RigidTransform rigidTransformFromSums(const PairSums& sums);

// Best-fit R and t that move source.points[i] onto target.points[matches[i]].
RigidTransform estimateRigidTransform(const Cloud& source, const Cloud& target, const std::vector<int>& matches);

// Angle in degrees of the rotation that takes `estimated` to `truth`. 0 = identical.
double rotationErrorDegrees(const Eigen::Matrix3f& estimated, const Eigen::Matrix3f& truth);
