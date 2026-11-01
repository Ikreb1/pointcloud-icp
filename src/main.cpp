#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Dense>

constexpr int TEST_SIZE = 10000;

// Max points per k-d tree leaf. Tweak and benchmark: smaller = deeper tree and
// more node visits, larger = longer brute-force scans in each leaf.
constexpr int KD_LEAF_SIZE = 16;

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

inline float coord(const Point& p, int axis)
{
	return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

// k-d tree for nearest-neighbour queries. Built once per target cloud (the
// target never moves during ICP), then queried for every source point.
//
// Layout: nodes live in one flat vector and refer to children by index. The
// points are copied and reordered so each leaf's points are contiguous in
// memory; `originalIndex` maps them back to indices in the input cloud.
class KdTree
{
public:
	explicit KdTree(const Cloud& cloud, int leafSize = KD_LEAF_SIZE)
		: m_leafSize(leafSize)
	{
		std::vector<int> order(cloud.points.size());
		for (int i = 0; i < static_cast<int>(order.size()); ++i)
			order[i] = i;

		m_nodes.reserve(2 * order.size() / leafSize + 1);
		build(cloud, order, 0, static_cast<int>(order.size()));

		m_points.reserve(order.size());
		for (int i : order)
			m_points.push_back(cloud.points[i]);
		m_originalIndex = std::move(order);
	}

	NearestResult nearest(const Point& query) const
	{
		NearestResult best;
		search(0, query, best);
		return best;
	}

	int nodeCount() const { return static_cast<int>(m_nodes.size()); }

private:
	struct Node
	{
		int axis;    // 0/1/2 = split axis, -1 = leaf
		float split; // interior: points with coord <= split go left, >= split go right
		int a, b;    // interior: left/right child node. leaf: [a, b) range in m_points
	};

	// Builds the subtree for order[begin, end) and returns its node index.
	int build(const Cloud& cloud, std::vector<int>& order, int begin, int end)
	{
		const int nodeIndex = static_cast<int>(m_nodes.size());
		m_nodes.push_back({});

		if (end - begin <= m_leafSize)
		{
			m_nodes[nodeIndex] = {-1, 0.0f, begin, end};
			return nodeIndex;
		}

		// Split on the axis where this subset is widest.
		Eigen::Vector3f lo = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
		Eigen::Vector3f hi = -lo;
		for (int i = begin; i < end; ++i)
		{
			const Point& p = cloud.points[order[i]];
			const Eigen::Vector3f v(p.x, p.y, p.z);
			lo = lo.cwiseMin(v);
			hi = hi.cwiseMax(v);
		}
		int axis;
		(hi - lo).maxCoeff(&axis);

		// Median split: nth_element puts the median at `mid`, smaller-or-equal
		// before it, larger-or-equal after it. O(n), no full sort needed.
		const int mid = begin + (end - begin) / 2;
		std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
			[&](int l, int r) { return coord(cloud.points[l], axis) < coord(cloud.points[r], axis); });
		const float split = coord(cloud.points[order[mid]], axis);

		// Children are built after this node was pushed, so m_nodes may have
		// reallocated: write through the index, never hold a reference across build().
		const int left = build(cloud, order, begin, mid);
		const int right = build(cloud, order, mid, end);
		m_nodes[nodeIndex] = {axis, split, left, right};
		return nodeIndex;
	}

	void search(int nodeIndex, const Point& query, NearestResult& best) const
	{
		const Node& node = m_nodes[nodeIndex];

		if (node.axis < 0)
		{
			// Leaf: brute force over its contiguous points.
			for (int i = node.a; i < node.b; ++i)
			{
				const Point& p = m_points[i];
				const float dx = p.x - query.x;
				const float dy = p.y - query.y;
				const float dz = p.z - query.z;
				const float d2 = dx * dx + dy * dy + dz * dz;
				if (d2 < best.distanceSquared)
				{
					best.distanceSquared = d2;
					best.index = m_originalIndex[i];
				}
			}
			return;
		}

		// Descend into the side the query is on first: it most likely holds
		// the nearest point, which shrinks `best` and lets us prune more.
		const float diff = coord(query, node.axis) - node.split;
		const int nearChild = diff < 0 ? node.a : node.b;
		const int farChild = diff < 0 ? node.b : node.a;

		search(nearChild, query, best);

		// Backtrack: everything on the far side is at least |diff| away along
		// this axis. Only search it if that could beat the best so far.
		if (diff * diff < best.distanceSquared)
			search(farChild, query, best);
	}

	int m_leafSize;
	std::vector<Node> m_nodes;
	std::vector<Point> m_points;
	std::vector<int> m_originalIndex;
};

enum class Search
{
	BruteForce,
	KdTree,
};

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

struct IcpResult
{
	RigidTransform transform; // accumulated over all iterations: source -> target
	int iterations = 0;
	float rmse = 0;          // root mean square distance between matched pairs
	double msPerIteration = 0;
	double buildMs = 0;      // k-d tree build, once per run
};

// Point-to-point ICP: repeatedly match each source point to its nearest target
// point, solve for the best rigid transform, and apply it. Stops when the error
// improves by less than `tolerance` or after `maxIterations`.
IcpResult runIcp(const Cloud& source, const Cloud& target, Search search, int maxIterations, float tolerance, bool verbose)
{
	IcpResult result;

	// The target never moves, so the tree is built once and reused every iteration.
	const auto buildStart = std::chrono::steady_clock::now();
	std::optional<KdTree> tree;
	if (search == Search::KdTree)
		tree.emplace(target);
	result.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();

	double totalMs = 0;
	Cloud current = source;
	std::vector<int> matches(source.points.size());
	float previousRmse = std::numeric_limits<float>::max();

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		const auto start = std::chrono::steady_clock::now();

		// Match: nearest target point for every source point.
		// Brute force is O(N*M), the tree roughly O(N log M).
		double sumSquared = 0;
		for (int i = 0; i < static_cast<int>(current.points.size()); ++i)
		{
			const NearestResult nearest = tree ? tree->nearest(current.points[i]) : nearestNeighbour(target, current.points[i]);
			matches[i] = nearest.index;
			sumSquared += nearest.distanceSquared;
		}
		const float rmse = static_cast<float>(std::sqrt(sumSquared / current.points.size()));

		// Align and apply.
		const RigidTransform step = estimateRigidTransform(current, target, matches);
		current = transformCloud(current, step.R, step.t);

		// Compose: total = step after total, i.e. p -> R_s (R p + t) + t_s.
		result.transform.R = step.R * result.transform.R;
		result.transform.t = step.R * result.transform.t + step.t;

		const auto end = std::chrono::steady_clock::now();
		const double ms = std::chrono::duration<double, std::milli>(end - start).count();
		totalMs += ms;
		if (verbose)
			std::cout << "iter " << iter << "  rmse " << rmse << "  " << ms << " ms\n";

		result.iterations = iter + 1;
		result.rmse = rmse;
		if (previousRmse - rmse < tolerance)
			break;
		previousRmse = rmse;
	}
	result.msPerIteration = totalMs / result.iterations;
	return result;
}

// Returns a copy of `in` with Gaussian noise (standard deviation `sigma`) added to every coordinate.
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

// Angle in degrees of the rotation that takes `estimated` to `truth`. 0 = identical.
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

void runScenario(const char* name, const Cloud& source, Search search, float angleDegrees, float noiseSigma, std::mt19937& rng)
{
	const float angle = angleDegrees * std::numbers::pi_v<float> / 180.0f;
	const Eigen::Matrix3f R = rotationZ(angle);
	const Eigen::Vector3f t(0.1f, -0.05f, 0.2f);

	Cloud target = transformCloud(source, R, t);
	if (noiseSigma > 0)
		target = addNoise(target, noiseSigma, rng);

	const IcpResult icp = runIcp(source, target, search, 100, 1e-6f, false);
	const double rotErr = rotationErrorDegrees(icp.transform.R, R);
	const float transErr = (icp.transform.t - t).norm();

	std::cout << name
	          << "  iters " << icp.iterations
	          << "  rmse " << icp.rmse
	          << "  rot err " << rotErr << " deg"
	          << "  trans err " << transErr
	          << "  " << icp.msPerIteration << " ms/iter";
	if (search == Search::KdTree)
		std::cout << " (+" << icp.buildMs << " ms build)";
	std::cout << (rotErr < 0.5 ? "  OK" : "  FAILED") << "\n";
}

// The tree must give exactly the same distances as brute force. Indices can
// differ only on exact ties, so compare distances.
bool kdTreeMatchesBruteForce(const Cloud& cloud, int queries, std::mt19937& rng)
{
	const KdTree tree(cloud);
	std::uniform_real_distribution<float> dist(-1.2f, 1.2f); // some queries outside the cloud
	for (int q = 0; q < queries; ++q)
	{
		const Point query{dist(rng), dist(rng), dist(rng)};
		const NearestResult expected = nearestNeighbour(cloud, query);
		const NearestResult actual = tree.nearest(query);
		if (actual.distanceSquared != expected.distanceSquared)
		{
			std::cout << "MISMATCH at query " << q << ": tree " << actual.distanceSquared << " vs brute " << expected.distanceSquared << "\n";
			return false;
		}
	}
	return true;
}

int main()
{
	std::mt19937 rng(42);
	const Cloud a = makeRandomCloud(TEST_SIZE, rng);

	std::cout << "k-d tree vs brute force, 10000 random queries: "
	          << (kdTreeMatchesBruteForce(a, 10000, rng) ? "identical" : "DIFFERENT") << "\n\n";

	std::cout << TEST_SIZE << " points, leaf size " << KD_LEAF_SIZE << "\n";
	runScenario("brute  10 deg, clean", a, Search::BruteForce, 10, 0.0f, rng);
	runScenario("kdtree 10 deg, clean", a, Search::KdTree, 10, 0.0f, rng);
	runScenario("kdtree 10 deg, noise", a, Search::KdTree, 10, 0.005f, rng);
	runScenario("kdtree 30 deg, clean", a, Search::KdTree, 30, 0.0f, rng);
	runScenario("kdtree 45 deg, clean", a, Search::KdTree, 45, 0.0f, rng);
	runScenario("kdtree 90 deg, clean", a, Search::KdTree, 90, 0.0f, rng);

	// Brute force at this size would be ~7 s per iteration.
	const Cloud big = makeRandomCloud(100000, rng);
	std::cout << "\n100000 points\n";
	runScenario("kdtree 10 deg, clean", big, Search::KdTree, 10, 0.0f, rng);
	runScenario("kdtree 10 deg, noise", big, Search::KdTree, 10, 0.005f, rng);
}
