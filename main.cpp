#include <cmath>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <vector_types.h>

#include "common.h"

using namespace std;

struct ElapsedTimer
{
	chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

	double elapsedSec()
	{
		chrono::duration<double> dur = chrono::high_resolution_clock::now() - start;
		return dur.count();
	}
};

vector<geom::Plane> readPlanesTxt(const string &filePath)
{
	vector<geom::Plane> inPlanes;
	inPlanes.reserve(1000000);

	ifstream file(filePath);
	for (double a, b, c, d; file >> a >> b >> c >> d;) {
		inPlanes.emplace_back(geom::Plane{
		    0,
		    static_cast<geom::real>(a),
		    static_cast<geom::real>(b),
		    static_cast<geom::real>(c),
		    static_cast<geom::real>(d)});
	}

	return inPlanes;
}

vector<geom::Plane> readPlanesBin(const string &filePath)
{
	// Assume data stores as list of double.
	// Each 4 double are 1 Plane(a, b, c, d).

	// Open the binary file
	ifstream file(filePath, ios::binary);

	if (!file) {
		cerr << "Error opening file: " << filePath << endl;
		return {};
	}

	streamsize fileSize = [&file] {
		file.seekg(0, ios::end);
		auto size = file.tellg();
		file.seekg(0, ios::beg);
		return size;
	}();

	size_t num_elements = fileSize / sizeof(double);
	vector<double> data(num_elements);
	file.read(reinterpret_cast<char *>(data.data()), fileSize);

	vector<geom::Plane> inPlanes;

	// Convert to Planes
	inPlanes.reserve(data.size() / 4);
	for (size_t i = 0; i < data.size(); i += 4) {
		inPlanes.push_back(geom::Plane{
		    0,
		    static_cast<geom::real>(data[i]),
		    static_cast<geom::real>(data[i + 1]),
		    static_cast<geom::real>(data[i + 2]),
		    static_cast<geom::real>(data[i + 3])});
	}

	return inPlanes;
}

void increasePlanesAmount(vector<geom::Plane> &planes)
{
	for (int i = 0; i < 8; ++i) {
		planes.insert(planes.end(), planes.begin(), planes.end());
	}
	planes.resize(80000000);
}

geom::PlaneList serialApproach(
    const geom::PlaneList &inPlanes,
    const geom::BasisList &basis_in,
    const geom::BasisList &basis_out)
{
	geom::PlaneList result;
	result.reserve(inPlanes.size());

	ElapsedTimer timer;
	for (size_t i = 0; i < inPlanes.size(); ++i) {
		// result.push_back(geom::getPolypointPlane(inPlanes[i], basis_in, basis_out));
	}
	double elapsed = timer.elapsedSec();

	ostringstream oss;
	oss << "Serial. Deformation took: " << elapsed << "\n";
	cout << oss.str();

	return result;
}

geom::PlaneList collectThreadResults(
    const geom::PlaneList &inPlanes, const vector<geom::PlaneList> &threadResults)
{
	geom::PlaneList result;
	result.reserve(inPlanes.size());
	for (size_t i = 0; i < threadResults.size(); ++i) {
		result.insert(result.end(), threadResults[i].begin(), threadResults[i].end());
	}
	return result;
}

geom::PlaneList threadChunkApproach(
    const geom::PlaneList &inPlanes,
    const geom::BasisList &basis_in,
    const geom::BasisList &basis_out,
    size_t threadCount)
{
	vector<thread> threads;
	size_t chunkSize = inPlanes.size() / threadCount;

	vector<geom::PlaneList> threadResults(threadCount);

	for (size_t i = 0; i < threadCount; ++i) {
		size_t startIdx = i * chunkSize;
		size_t endIdx = (i == threadCount - 1) ? inPlanes.size() : startIdx + chunkSize;

		threads.push_back(thread([&, startIdx, endIdx, i] {
			for (size_t j = startIdx; j < endIdx; ++j) {
				// threadResults[i].push_back(
				// geom::getPolypointPlane(inPlanes[j], basis_in, basis_out));
			}
		}));
	}

	for (size_t i = 0; i < threads.size(); ++i) {
		threads[i].join();
	}

	return collectThreadResults(inPlanes, threadResults);
}

int main(int argc, char *argv[])
{
	const int runEachExperiment = 1;

	char cwd[256];
	cout << "Working dir " << _getcwd(cwd, sizeof(cwd)) << "\n";
	string planesFile = "./in_planes.npy";
	cout << "Planes file: " << planesFile << "\n";

	ElapsedTimer timer;
	vector<geom::Plane> inPlanes = readPlanesBin(planesFile);
	increasePlanesAmount(inPlanes);

	// clang-format off
	ostringstream oss;
	oss << "Read " << inPlanes.size() << " planes in " << timer.elapsedSec() << " seconds\n";
	oss << "Last plane: " << inPlanes.back().a << " " << inPlanes.back().b << " " << inPlanes.back().c << " " << inPlanes.back().d << "\n";
	oss << "Data Type: " << typeid(geom::real).name() << "\n\n";
	cout << oss.str();

	auto basis_in = geom::BasisList{
		{0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, -0.0},
		{1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, -0.0} };
	auto basis_out = geom::BasisList{
		{0.0, 0.0, 1.0},  {0.2, 0.2,   1.0}, {0.0, 0.0,      0.0}, {0.0, 1.0, -0.0},
		{0.2, -0.2, 1.0}, {1.18, 0.78, 1.0}, {1.0, 6.12e-17, 0.0}, {1.0, 1.0, -0.0} };
	// clang-format on

	{
		vector<double> times;

		for (int i = 0; i < runEachExperiment; ++i) {
			ElapsedTimer t;
			geom::PlaneList result;
			gpu::deformPlanesPolypoint(result, inPlanes, basis_in, basis_out);
			times.push_back(t.elapsedSec());

			// Print result
			// for (const auto &p : std::vector(result.begin() + 10000, result.begin() + 10000 +
			// 200)) { 	std::cout << std::fixed << std::setprecision(50) << p.a << ";" << p.b << ";"
			//<< p.c
			//	          << ";" << p.d << "\n";
			//}
		}

		double avg = accumulate(times.begin(), times.end(), 0.0) / runEachExperiment;
		cout << "Average of " << runEachExperiment << " run time: " << avg << " seconds\n";
	}

	return 0;
}
