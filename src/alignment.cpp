#include "alignment.h"

#include <cmath>
#include <numbers>

RigidTransform rigidTransformFromSums(const PairSums& sums)
{
	// 1. Centroids: the average source point and the average target point.
	const Eigen::Vector3d sourceCentroid = sums.sourceSum / sums.count;
	const Eigen::Vector3d targetCentroid = sums.targetSum / sums.count;

	// 2 + 3. Cross-covariance of the centred pairs, H = sum of (s - sourceCentroid)(d - targetCentroid)^T.
	// Multiplying that out gives the same H from the raw sums, in one pass:
	//   H = outerSum - count * sourceCentroid * targetCentroid^T
	// Caveat: this subtracts two large, nearly equal matrices when the clouds sit
	// far from the origin, which loses digits (catastrophic cancellation).
	// Fine here: data is near the origin and the sums are double.
	const Eigen::Matrix3d H = sums.outerSum - static_cast<double>(sums.count) * sourceCentroid * targetCentroid.transpose();

	// 4. H = U * S * V^T. The closest rotation is R = V * U^T.
	const Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(H);
	const Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d R = V * U.transpose();

	// det = -1 means a reflection, not a rotation: flip the axis with the
	// smallest singular value (SVD sorts them, so it is the last column).
	if (R.determinant() < 0)
	{
		V.col(2) *= -1;
		R = V * U.transpose();
	}

	// 5. Rotate the source centre, then shift it onto the target centre.
	const Eigen::Vector3d t = targetCentroid - R * sourceCentroid;

	return {R.cast<float>(), t.cast<float>()};
}

RigidTransform estimateRigidTransform(const Cloud& source, const Cloud& target, const std::vector<int>& matches)
{
	PairSums sums;
	for (int i = 0; i < static_cast<int>(source.points.size()); ++i)
		sums.add(source.points[i], target.points[matches[i]]);
	return rigidTransformFromSums(sums);
}

// Uses atan2(sin, cos) instead of acos(cos): float matrices are never exactly
// orthonormal, and acos near 1 turns that ~1e-7 rounding into ~0.03 degrees of
// fake error. The sin part comes from the antisymmetric half of the matrix,
// which only changes when the rotation really differs.
double rotationErrorDegrees(const Eigen::Matrix3f& estimated, const Eigen::Matrix3f& truth)
{
	const Eigen::Matrix3d D = estimated.cast<double>().transpose() * truth.cast<double>();
	const double sinAngle = 0.5 * Eigen::Vector3d(D(2, 1) - D(1, 2), D(0, 2) - D(2, 0), D(1, 0) - D(0, 1)).norm();
	const double cosAngle = (D.trace() - 1.0) / 2.0;
	return std::atan2(sinAngle, cosAngle) * 180.0 / std::numbers::pi;
}
