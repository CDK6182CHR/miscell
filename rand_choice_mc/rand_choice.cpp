#include <random>
#include <bitset>
#include <assert.h>
#include <array>
#include <memory>
#include <chrono>
#include <iostream>
#include <type_traits>

consteval int ipow(int x, int n) {
	if (n == 0) {
		return 1;
	}
	else if (n < 0) {
		return 0;
	}
	else if (n == 1) {
		return x;
	}
	else {
		return x * ipow(x, n - 1);
	}
}


template<size_t _N>
int cal_score(const std::bitset<_N>& a, const std::bitset<_N>& b)
{
	auto t1 = a & b;
	if (t1.none()) {
		return 0;
	}
	else if (t1.all()) {
		return _N;
	}

	auto t2 = a ^ b;
	return _N - t2.count();
}


template <size_t _N>
class BaseGenerator {

};

template <size_t _N, typename _Ty>
concept ChoiceGen = std::is_base_of_v<BaseGenerator<_N>, _Ty>;

template <size_t _N>
class IndependentGenerator : public BaseGenerator<_N> {
	std::uniform_real_distribution<> dist;
	std::default_random_engine gen;
	double _rate{ 0.5 };

public:
	template <typename ...Args>
	IndependentGenerator(Args&&... args) :
		dist(0, 1), gen(std::forward<Args>(args)...) {}

	std::bitset<_N> generate();

	void set_rate(double r) { _rate = r; }

private:
	std::bitset<_N> simple_generate();
};

template<size_t _N>
std::bitset<_N> IndependentGenerator<_N>::generate()
{
	std::bitset<_N> res = simple_generate();
	while (res.none())res = simple_generate();
	return res;
}

template<size_t _N>
std::bitset<_N> IndependentGenerator<_N>::simple_generate()
{
	std::bitset<_N> res;
	res.reset();
	for (int i = 0; i < _N; i++) {
		if (double t = dist(gen); t < _rate) {
			res.set(i);
		}
	}
	return res;
}



template <size_t _N>
class CombinationGenerator :public BaseGenerator<_N> {
	static_assert(_N < sizeof(unsigned long long) * 8, "Invalid size");
	//constexpr int CHOICES = ipow(2, _N);

	std::uniform_int_distribution<> dist;
	std::default_random_engine gen;

public:
	template <typename... Args>
	CombinationGenerator(Args&&...args) :
		dist(1, ipow(2, _N)-1), gen(std::forward<Args>(args)...) {}

	std::bitset<_N> generate();
};

template<size_t _N>
std::bitset<_N> CombinationGenerator<_N>::generate()
{
	return std::bitset<_N>(dist(gen));
}



template <size_t _N>
class CountGenerator :public BaseGenerator<_N> {
	std::uniform_int_distribution<> dist;
	std::default_random_engine gen;
	int _count;
public:
	template <typename... Args>
	CountGenerator(int count, Args&&... args) :
		dist(0, _N - 1), gen(std::forward<Args>(args)...), _count(count) {
	}

	std::bitset<_N> generate();
};

template <size_t _N>
std::bitset<_N> CountGenerator<_N>::generate()
{
	std::bitset<_N> res;
	if (_count == _N) {
		return res.set();
	}
	else if (_count > _N / 2) {
		res.set();
		int t = _N - _count;
		while (t > 0) {
			int i = dist(gen);
			if (res.test(i)) {
				res.reset(i);
				t--;
			}
		}
	}
	else {
		res.reset();
		int t = _count;
		while (t > 0) {
			int i = dist(gen);
			if (!res.test(i)) {
				res.set(i);
				t--;
			}
		}
	}
	return res;
}



template <size_t _N, typename _Gen>
requires ChoiceGen<_N,_Gen>
std::array<double, _N> mc_single(int total, _Gen& gen)
{
	auto clk_start = std::chrono::system_clock::now();

	std::random_device r;
	std::array<double, _N> res{};
	std::array<std::unique_ptr<CountGenerator<_N>>, _N> ans_gens;
	for (int i = 0; i < _N; i++) {
		ans_gens[i] = std::make_unique<CountGenerator<_N>>(i + 1, r());
	}

	for (int i = 0; i < total; i++) {
		auto std_ans = gen.generate();
		for (int k = 0; k < _N; k++) {
			auto choice = ans_gens.at(k)->generate();
			int s = cal_score(std_ans, choice);
			res.at(k) += s;
		}
	}

	for (int k = 0; k < _N; k++) {
		res.at(k) /= total;
	}

	auto clk_end = std::chrono::system_clock::now();
	using namespace std::chrono_literals;
	std::cout << "Simulation for " << total << " trials takes "
		<< (clk_end - clk_start) / 1ms << " ms" << std::endl;

	return res;
}


#include <fstream>

template <size_t _N, typename _Gen>
requires ChoiceGen<_N,_Gen>
void converge_test(_Gen& gen)
{
	std::ofstream fout("converge.txt", std::ios::out);
	fout << "cycles ";
	for (int i = 1; i <= _N; i++) {
		fout << i << " ";
	}
	fout << "\n";

	int base = 10000;
	for (int i = 1; i <= 50; i++) {
		int tot = base * i;
		auto res = mc_single<_N>(tot, gen);
		std::cout << tot << ' ';
		fout << tot << ' ';
		for (const auto& d : res) {
			std::cout << d << ' ';
			fout << d << ' ';
		}
		std::cout << '\n';
		fout << '\n';
	}
}

template <size_t _N>
void indep_rate_test(int total)
{
	std::ofstream fout("rate.txt", std::ios::out);
	fout << "rate ";
	for (int i = 1; i <= _N; i++) {
		fout << i << ' ';
	}
	fout << std::endl;

	IndependentGenerator<_N> gen(time(NULL));

	for (double x = 0.15; x <= 0.86; x += 0.05) {
		gen.set_rate(x);
		auto res = mc_single<_N>(total, gen);
		fout << x << ' ';
		for (double t : res) {
			fout << t << ' ';
		}
		fout << std::endl;
	}
	fout.close();
}


#include <ctime>

int main()
{
	// CombinationGenerator<4> gen(time(NULL));
	// converge_test<4>(gen);
	indep_rate_test<4>(500000);
}


