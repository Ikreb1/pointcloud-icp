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
}
