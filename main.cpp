#include <iostream>
#include <cmath>
#include <functional>
#include <chrono>
#include <omp.h>
#include <cassert>

#define _USE_MATH_DEFINES // Фикс для M_PI
#include <cmath>

constexpr size_t THREADS = 4;
constexpr size_t N = 10'000'000;
constexpr double EPS = 1e-8;

// Функция для интегрирования
double f(double x)
{
	return std::exp(-std::pow(x, 2));
}

// Последовательное интегрирование
double integrate(std::function<double(double)> f, double a, double b)
{
	double r = 0.0;
	double h = (b - a) / N;
	for (size_t i = 1; i < N; i++)
	{
		double x = a + (i - 0.5) * h; // Центр прямоугольника
		r += f(x);
	}
	return r * h;
}

// Параллельное интегрирование
double integrate_parallel(std::function<double(double)> f, double a, double b)
{
	double r = 0.0;
	double h = (b - a) / N;

#pragma omp parallel for reduction(+ : r)
	for (size_t i = 1; i < N; i++)
	{
		double x = a + (i - 0.5) * h;
		r += f(x);
	}
	return r * h;
}

int main()
{
	omp_set_num_threads(THREADS);

	auto start_serial = std::chrono::steady_clock::now();
	double r1 = integrate(f, -M_PI, M_PI);
	auto end_serial = std::chrono::steady_clock::now();

	auto start_parallel = std::chrono::steady_clock::now();
	double r2 = integrate_parallel(f, -M_PI, M_PI);
	auto end_parallel = std::chrono::steady_clock::now();

	assert(std::fabs(r1 - r2) < EPS);

	std::cout << "RESULT: " << r1 << std::endl;
	std::cout << "PRECISE: " << std::sqrt(M_PI) * std::erf(M_PI) << std::endl;
	std::cout << "SERIAL: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial).count()
			  << "ms" << std::endl;
	std::cout << "PARALLEL: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count()
			  << "ms" << std::endl;

	return 0;
}
