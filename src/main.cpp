#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Dense>

constexpr int TEST_SIZE = 10000;

struct Point
{
	float x = 0, y = 0, z = 0;

	void randomize(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		x = dist(rng);
		y = dist(rng);
		z = dist(rng);
	}
};

struct Cloud
{
	std::vector<Point> points;
};

Cloud makeRandomCloud(int count, std::mt19937& rng)
{
	Cloud cloud;
	cloud.points.reserve(count);
	for (int i = 0; i < count; ++i)
	{
		Point p;
		p.randomize(rng);
		cloud.points.push_back(p);
	}
	return cloud;
}

// Rotation of `radians` around the z axis.
Eigen::Matrix3f rotationZ(float radians)
{
	const float c = std::cos(radians);
	const float s = std::sin(radians);

	Eigen::Matrix3f R;
	R << c, -s, 0,
	     s,  c, 0,
	     0,  0, 1;
	return R;
}

// Returns a copy of `in` with p' = R * p + t applied to every point.
Cloud transformCloud(const Cloud& in, const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
	Cloud out;
	out.points.reserve(in.points.size());
	for (const Point& p : in.points)
	{
		const Eigen::Vector3f q = R * Eigen::Vector3f(p.x, p.y, p.z) + t;
		out.points.push_back({q.x(), q.y(), q.z()});
	}
	return out;
}

struct NearestResult
{
	int index = -1;
	float distanceSquared = std::numeric_limits<float>::max();
};

// Brute force: checks every point in `target`. Compares squared distances,
// since sqrt preserves order and is expensive.
NearestResult nearestNeighbour(const Cloud& target, const Point& query)
{
	NearestResult best;
	for (int i = 0; i < static_cast<int>(target.points.size()); ++i)
	{
		const Point& p = target.points[i];
		const float dx = p.x - query.x;
		const float dy = p.y - query.y;
		const float dz = p.z - query.z;
		const float d2 = dx * dx + dy * dy + dz * dz;
		if (d2 < best.distanceSquared)
		{
			best.distanceSquared = d2;
			best.index = i;
		}
	}
	return best;
}

struct RigidTransform
{
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

// Best-fit R and t that move source.points[i] onto target.points[matches[i]]
// (least squares, Kabsch/SVD method). Sums are done in double: adding
// thousands of floats into one running total loses precision.
RigidTransform estimateRigidTransform(const Cloud& source, const Cloud& target, const std::vector<int>& matches)
{
	const int n = static_cast<int>(source.points.size());

	// 1. Centroids.
	Eigen::Vector3d sourceCentroid = Eigen::Vector3d::Zero();
	Eigen::Vector3d targetCentroid = Eigen::Vector3d::Zero();
	for (int i = 0; i < n; ++i)
	{
		const Point& s = source.points[i];
		const Point& d = target.points[matches[i]];
		sourceCentroid += Eigen::Vector3d(s.x, s.y, s.z);
		targetCentroid += Eigen::Vector3d(d.x, d.y, d.z);
	}
	sourceCentroid /= n;
	targetCentroid /= n;

	// 2 + 3. Cross-covariance of the centred pairs: H = sum of s * d^T.
	Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
	for (int i = 0; i < n; ++i)
	{
		const Point& s = source.points[i];
		const Point& d = target.points[matches[i]];
		const Eigen::Vector3d sc = Eigen::Vector3d(s.x, s.y, s.z) - sourceCentroid;
		const Eigen::Vector3d dc = Eigen::Vector3d(d.x, d.y, d.z) - targetCentroid;
		H += sc * dc.transpose();
	}

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

void printPoint(const char* label, const Point& p)
{
	std::cout << label << " (" << p.x << ", " << p.y << ", " << p.z << ")\n";
}

int main()
{
	std::mt19937 rng(42);
	Cloud a = makeRandomCloud(TEST_SIZE, rng);

	const float angle = 10.0f * std::numbers::pi_v<float> / 180.0f;
	const Eigen::Matrix3f R = rotationZ(angle);
	const Eigen::Vector3f t(0.1f, -0.05f, 0.2f);

	Cloud b = transformCloud(a, R, t);

	std::cout << "R =\n" << R << "\n";
	std::cout << "det(R) = " << R.determinant() << "\n";
	std::cout << "R^T R is identity: " << std::boolalpha << (R.transpose() * R).isIdentity(1e-6f) << "\n";
	printPoint("a[0] =", a.points[0]);
	printPoint("b[0] =", b.points[0]);

	const NearestResult self = nearestNeighbour(a, a.points[5]);
	std::cout << "nearest to a[5] in a: index " << self.index << ", d2 " << self.distanceSquared << "\n";

	const NearestResult moved = nearestNeighbour(a, b.points[5]);
	std::cout << "nearest to b[5] in a: index " << moved.index << ", d2 " << moved.distanceSquared << "\n";

	// With perfect matches (a[i] became b[i]) the alignment must recover R and t exactly.
	std::vector<int> perfectMatches(a.points.size());
	for (int i = 0; i < static_cast<int>(perfectMatches.size()); ++i)
		perfectMatches[i] = i;

	const RigidTransform recovered = estimateRigidTransform(a, b, perfectMatches);
	std::cout << "recovered R =\n" << recovered.R << "\n";
	std::cout << "recovered t = " << recovered.t.transpose() << "\n";
	std::cout << "R error = " << (recovered.R - R).norm() << ", t error = " << (recovered.t - t).norm() << "\n";
}
