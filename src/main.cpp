#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <immintrin.h>
#include <omp.h>

constexpr int TEST_SIZE = 10000;

// Max points per k-d tree leaf. Tweak and benchmark: smaller = deeper tree and
// more node visits, larger = longer brute-force scans in each leaf.
constexpr int KD_LEAF_SIZE = 16;
static_assert(KD_LEAF_SIZE % 8 == 0, "leaves must be whole 8-wide SIMD registers");

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

// Applies p' = R * p + t to every point of `cloud`, in place and in parallel.
// Used inside ICP: no new allocation each iteration, and each point is independent.
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

struct NearestResult
{
	int index = -1;
	float distanceSquared = std::numeric_limits<float>::max();
};

// Brute force: checks every point in `target`. Compares squared distances,
// since sqrt preserves order and is expensive.
// noinline: keeps brute force as its own entry in perf report instead of being
// folded into runIcp. One call per 10k distance checks, so the cost is negligible.
[[gnu::noinline]] NearestResult nearestNeighbour(const Cloud& target, const Point& query)
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

#ifdef __AVX2__
// Smallest of the 8 floats in a 256-bit register ("horizontal min").
inline float horizontalMin(__m256 v)
{
	__m128 m = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); // 8 -> 4
	m = _mm_min_ps(m, _mm_movehl_ps(m, m));                                          // 4 -> 2
	m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));                                      // 2 -> 1
	return _mm_cvtss_f32(m);
}
#endif

// k-d tree for nearest-neighbour queries. Built once per target cloud (the
// target never moves during ICP), then queried for every source point.
//
// Layout:
// - Nodes live in one flat vector and refer to children by index.
// - Points are stored as Structure of Arrays (separate x, y, z arrays),
//   reordered so each leaf's points are contiguous. One AVX2 load then
//   fills a register with 8 x values of the same leaf.
// - Every leaf starts at a multiple of 8 and holds a multiple of 8 points
//   (except the very last leaf), so leaves are whole SIMD registers. The
//   arrays are padded to a multiple of 8 with points at infinity, which are
//   never the nearest.
class KdTree
{
public:
	explicit KdTree(const Cloud& cloud, int leafSize = KD_LEAF_SIZE)
		: m_leafSize(leafSize)
	{
		const int n = static_cast<int>(cloud.points.size());
		std::vector<int> order(n);
		for (int i = 0; i < n; ++i)
			order[i] = i;

		m_nodes.reserve(2 * n / leafSize + 1);
		build(cloud, order, 0, n);

		const int padded = (n + SIMD_WIDTH - 1) / SIMD_WIDTH * SIMD_WIDTH;
		const float inf = std::numeric_limits<float>::infinity();
		m_x.assign(padded, inf);
		m_y.assign(padded, inf);
		m_z.assign(padded, inf);
		m_originalIndex.assign(padded, -1);
		for (int i = 0; i < n; ++i)
		{
			const Point& p = cloud.points[order[i]];
			m_x[i] = p.x;
			m_y[i] = p.y;
			m_z[i] = p.z;
			m_originalIndex[i] = order[i];
		}
	}

	// Leaf scans with AVX2: 8 points per instruction.
	NearestResult nearest(const Point& query) const
	{
		NearestResult best;
		search<true>(0, query, best);
		return best;
	}

	// Same tree and layout, leaf scans one point at a time. For benchmarking.
	NearestResult nearestScalar(const Point& query) const
	{
		NearestResult best;
		search<false>(0, query, best);
		return best;
	}

	static constexpr int SIMD_WIDTH = 8; // floats per 256-bit AVX2 register

private:
	struct Node
	{
		int axis;    // 0/1/2 = split axis, -1 = leaf
		float split; // interior: points with coord <= split go left, >= split go right
		int a, b;    // interior: left/right child node. leaf: [a, b) range in the point arrays
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

		// Split near the median, rounded up to a multiple of 8 points. `begin`
		// starts at 0 and every left half is a multiple of 8, so every node starts
		// on a multiple of 8: leaves are whole SIMD registers.
		const int half = (end - begin) / 2;
		const int mid = begin + (half + SIMD_WIDTH - 1) / SIMD_WIDTH * SIMD_WIDTH;

		// nth_element puts the element that belongs at `mid` there, smaller-or-equal
		// before it, larger-or-equal after it. O(n), no full sort needed.
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

	void scanLeafScalar(const Node& node, const Point& query, NearestResult& best) const
	{
		for (int i = node.a; i < node.b; ++i)
		{
			const float dx = m_x[i] - query.x;
			const float dy = m_y[i] - query.y;
			const float dz = m_z[i] - query.z;
			const float d2 = dx * dx + dy * dy + dz * dz;
			if (d2 < best.distanceSquared)
			{
				best.distanceSquared = d2;
				best.index = m_originalIndex[i];
			}
		}
	}

#ifdef __AVX2__
	void scanLeafSimd(const Node& node, const Point& query, NearestResult& best) const
	{
		// Broadcast: the query's x copied into all 8 lanes, same for y and z.
		const __m256 qx = _mm256_set1_ps(query.x);
		const __m256 qy = _mm256_set1_ps(query.y);
		const __m256 qz = _mm256_set1_ps(query.z);

		// Blocks of 8 points. May read past node.b into the next leaf or the
		// infinity padding; both are harmless (real points are just extra
		// candidates, padding is never closer).
		for (int i = node.a; i < node.b; i += SIMD_WIDTH)
		{
			const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(&m_x[i]), qx);
			const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(&m_y[i]), qy);
			const __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(&m_z[i]), qz);
			__m256 d2 = _mm256_mul_ps(dx, dx);
			d2 = _mm256_fmadd_ps(dy, dy, d2); // d2 += dy * dy, in one instruction
			d2 = _mm256_fmadd_ps(dz, dz, d2);

			// Usually no point in the block beats the best so far: one compare
			// and move on. Only when one does, find which lane it was.
			const float blockMin = horizontalMin(d2);
			if (blockMin < best.distanceSquared)
			{
				const __m256 isMin = _mm256_cmp_ps(d2, _mm256_set1_ps(blockMin), _CMP_EQ_OQ);
				const int lane = std::countr_zero(static_cast<unsigned>(_mm256_movemask_ps(isMin)));
				best.distanceSquared = blockMin;
				best.index = m_originalIndex[i + lane];
			}
		}
	}
#endif

	template <bool UseSimd>
	void search(int nodeIndex, const Point& query, NearestResult& best) const
	{
		const Node& node = m_nodes[nodeIndex];

		if (node.axis < 0)
		{
#ifdef __AVX2__
			if constexpr (UseSimd)
			{
				scanLeafSimd(node, query, best);
				return;
			}
#endif
			scanLeafScalar(node, query, best);
			return;
		}

		// Descend into the side the query is on first: it most likely holds
		// the nearest point, which shrinks `best` and lets us prune more.
		const float diff = coord(query, node.axis) - node.split;
		const int nearChild = diff < 0 ? node.a : node.b;
		const int farChild = diff < 0 ? node.b : node.a;

		search<UseSimd>(nearChild, query, best);

		// Backtrack: everything on the far side is at least |diff| away along
		// this axis. Only search it if that could beat the best so far.
		if (diff * diff < best.distanceSquared)
			search<UseSimd>(farChild, query, best);
	}

	int m_leafSize;
	std::vector<Node> m_nodes;
	std::vector<float> m_x, m_y, m_z;
	std::vector<int> m_originalIndex;
};

enum class Search
{
	BruteForce,
	KdTreeScalar,
	KdTreeSimd,
};

struct RigidTransform
{
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

// Running totals over matched pairs (source point, target point). Everything
// the alignment needs fits in these 15 numbers plus a count, so they can be
// accumulated during matching, one copy per thread, and added together after.
// Sums are in double: adding thousands of floats into one running total loses precision.
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

// Best-fit R and t that move source.points[i] onto target.points[matches[i]].
RigidTransform estimateRigidTransform(const Cloud& source, const Cloud& target, const std::vector<int>& matches)
{
	PairSums sums;
	for (int i = 0; i < static_cast<int>(source.points.size()); ++i)
		sums.add(source.points[i], target.points[matches[i]]);
	return rigidTransformFromSums(sums);
}

struct IcpResult
{
	RigidTransform transform; // accumulated over all iterations: source -> target
	int iterations = 0;
	float rmse = 0;          // root mean square distance between matched pairs
	double msPerIteration = 0;
	double buildMs = 0;      // k-d tree build, once per run

	// Per-iteration averages of each phase, in ms.
	double matchMs = 0;      // nearest neighbour + pair sums for every point (parallel)
	double alignMs = 0;      // centroids, H and SVD from the pair sums (serial, tiny)
	double transformMs = 0;  // apply R, t to the moving cloud (parallel)
};

double msSince(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// Point-to-point ICP: repeatedly match each source point to its nearest target
// point, solve for the best rigid transform, and apply it. Stops when the error
// improves by less than `tolerance` or after `maxIterations`.
IcpResult runIcp(const Cloud& source, const Cloud& target, Search search, int maxIterations, float tolerance, bool verbose)
{
	IcpResult result;

	// The target never moves, so the tree is built once and reused every iteration.
	const auto buildStart = std::chrono::steady_clock::now();
	std::optional<KdTree> tree;
	if (search != Search::BruteForce)
		tree.emplace(target);
	result.buildMs = msSince(buildStart);

	double totalMs = 0, matchMs = 0, alignMs = 0, transformMs = 0;
	Cloud current = source;
	float previousRmse = std::numeric_limits<float>::max();

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		const auto start = std::chrono::steady_clock::now();
		auto phase = start;

		// Match: nearest target point for every source point, and add each pair
		// straight into the alignment sums (no separate pass over the pairs).
		// Brute force is O(N*M), the tree roughly O(N log M).
		// Threads only read the tree/target. The two totals are `reduction`s: each
		// thread fills its own private copy, OpenMP adds them up at the end.
		double sumSquared = 0;
		PairSums sums;
#pragma omp parallel for reduction(+ : sumSquared, sums) schedule(static)
		for (int i = 0; i < static_cast<int>(current.points.size()); ++i)
		{
			const Point& query = current.points[i];
			const NearestResult nearest = search == Search::KdTreeSimd ? tree->nearest(query)
			                            : search == Search::KdTreeScalar ? tree->nearestScalar(query)
			                            : nearestNeighbour(target, query);
			sumSquared += nearest.distanceSquared;
			sums.add(query, target.points[nearest.index]);
		}
		const float rmse = static_cast<float>(std::sqrt(sumSquared / current.points.size()));
		matchMs += msSince(phase);

		// Align.
		phase = std::chrono::steady_clock::now();
		const RigidTransform step = rigidTransformFromSums(sums);
		alignMs += msSince(phase);

		// Apply.
		phase = std::chrono::steady_clock::now();
		transformInPlace(current, step.R, step.t);
		transformMs += msSince(phase);

		// Compose: total = step after total, i.e. p -> R_s (R p + t) + t_s.
		result.transform.R = step.R * result.transform.R;
		result.transform.t = step.R * result.transform.t + step.t;

		const double ms = msSince(start);
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
	result.matchMs = matchMs / result.iterations;
	result.alignMs = alignMs / result.iterations;
	result.transformMs = transformMs / result.iterations;
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

	const IcpResult icp = runIcp(source, target, search, 500, 1e-6f, false);
	const double rotErr = rotationErrorDegrees(icp.transform.R, R);
	const float transErr = (icp.transform.t - t).norm();

	std::cout << name
	          << "  iters " << icp.iterations
	          << "  rmse " << icp.rmse
	          << "  rot err " << rotErr << " deg"
	          << "  trans err " << transErr
	          << "  " << icp.msPerIteration << " ms/iter";
	if (search != Search::BruteForce)
		std::cout << " (+" << icp.buildMs << " ms build)";
	std::cout << (rotErr < 0.5 ? "  OK" : "  FAILED") << "\n";

	const double pct = 100.0 / icp.msPerIteration;
	std::cout << "    per iter: match " << icp.matchMs << " ms (" << icp.matchMs * pct << "%)"
	          << "  align " << icp.alignMs << " ms (" << icp.alignMs * pct << "%)"
	          << "  transform " << icp.transformMs << " ms (" << icp.transformMs * pct << "%)\n";
}

// The tree must find the same nearest distances as brute force. Allows a
// rounding-sized difference: the SIMD path computes dx*dx + dy*dy + dz*dz with
// FMA in a fixed order, which can differ from the scalar code in the last bit.
// Indices can differ only on exact ties, so compare distances.
bool kdTreeMatchesBruteForce(const Cloud& cloud, Search search, int queries, std::mt19937& rng)
{
	const KdTree tree(cloud);
	std::uniform_real_distribution<float> dist(-1.2f, 1.2f); // some queries outside the cloud
	for (int q = 0; q < queries; ++q)
	{
		const Point query{dist(rng), dist(rng), dist(rng)};
		const NearestResult expected = nearestNeighbour(cloud, query);
		const NearestResult actual = search == Search::KdTreeSimd ? tree.nearest(query) : tree.nearestScalar(query);
		if (std::abs(actual.distanceSquared - expected.distanceSquared) > 1e-6f * expected.distanceSquared + 1e-12f)
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
	const Cloud small = makeRandomCloud(TEST_SIZE, rng);

	std::cout << omp_get_max_threads() << " threads, leaf size " << KD_LEAF_SIZE << "\n\n";

	std::cout << "Correctness vs brute force, 10000 random queries:\n";
	std::cout << "  k-d tree scalar: " << (kdTreeMatchesBruteForce(small, Search::KdTreeScalar, 10000, rng) ? "same" : "DIFFERENT") << "\n";
	std::cout << "  k-d tree SIMD:   " << (kdTreeMatchesBruteForce(small, Search::KdTreeSimd, 10000, rng) ? "same" : "DIFFERENT") << "\n\n";

	std::cout << "Accuracy, " << TEST_SIZE << " points:\n";
	runScenario("brute  10 deg, clean", small, Search::BruteForce, 10, 0.0f, rng);
	runScenario("simd   10 deg, clean", small, Search::KdTreeSimd, 10, 0.0f, rng);
	runScenario("simd   10 deg, noise", small, Search::KdTreeSimd, 10, 0.005f, rng);
	runScenario("simd   30 deg, clean", small, Search::KdTreeSimd, 30, 0.0f, rng);
	runScenario("simd   45 deg, clean", small, Search::KdTreeSimd, 45, 0.0f, rng);

	for (int size : {100000, 1000000})
	{
		const Cloud big = makeRandomCloud(size, rng);
		std::cout << "\nSpeed, " << size << " points, 10 deg + noise:\n";
		runScenario("scalar", big, Search::KdTreeScalar, 10, 0.005f, rng);
		runScenario("simd  ", big, Search::KdTreeSimd, 10, 0.005f, rng);
	}
}
