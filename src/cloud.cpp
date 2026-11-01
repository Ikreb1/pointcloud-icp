#include "cloud.h"

#include <cmath>

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

Cloud addNoise(const Cloud& in, float sigma, std::mt19937& rng)
{
	std::normal_distribution<float> noise(0.0f, sigma);
	Cloud out = in;
	for (Point& p : out.points)
	{
		p.x += noise(rng);
		p.y += noise(rng);
		p.z += noise(rng);
	}
	return out;
}

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

void transformInPlace(Cloud& cloud, const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < static_cast<int>(cloud.points.size()); ++i)
	{
		Point& p = cloud.points[i];
		const Eigen::Vector3f q = R * Eigen::Vector3f(p.x, p.y, p.z) + t;
		p = {q.x(), q.y(), q.z()};
	}
}
