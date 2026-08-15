#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using u128 = unsigned __int128;

static u64 mod_mul(u64 a, u64 b, u64 p) {
    return static_cast<u64>((static_cast<u128>(a) * b) % p);
}

static u64 mod_pow(u64 a, u64 e, u64 p) {
    u64 result = 1;
    while (e != 0) {
        if (e & 1) result = mod_mul(result, a, p);
        a = mod_mul(a, a, p);
        e >>= 1;
    }
    return result;
}

static inline void add_assign(u64 &target, u64 value, u64 p) {
    const u64 sum = target + value;
    target = sum >= p ? sum - p : sum;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 7) {
        std::cerr << "usage: moments_mod PRIME PRIMITIVE_ROOT MAX_MOMENT [DIMENSION MAG1_COUNT MAG2_COUNT]\n";
        return 2;
    }
    const u64 mod = std::stoull(argv[1]);
    const u64 primitive = std::stoull(argv[2]);
    const int max_moment = std::stoi(argv[3]);
    const int dimension = argc == 7 ? std::stoi(argv[4]) : 64;
    const int mag1_count = argc == 7 ? std::stoi(argv[5]) : 31;
    const int mag2_count = argc == 7 ? std::stoi(argv[6]) : 11;
    if (dimension <= 0 || mag1_count < 0 || mag2_count < 0 ||
        mag1_count + mag2_count > dimension || (mod - 1) % (2 * dimension) != 0) {
        std::cerr << "invalid dimension, shell counts, or modulus\n";
        return 2;
    }

    const int side = max_moment + 1;
    const int area = side * side;
    const int mag2_stride = area;
    const int mag1_stride = (mag2_count + 1) * area;
    auto offset = [mag1_stride, mag2_stride, side](int a, int b, int r, int s) {
        return a * mag1_stride + b * mag2_stride + r * side + s;
    };

    const u64 root = mod_pow(primitive, (mod - 1) / (2 * dimension), mod);
    if (mod_pow(root, 2 * dimension, mod) != 1 ||
        mod_pow(root, dimension, mod) != mod - 1) {
        std::cerr << "supplied generator did not produce a primitive 2d-th root\n";
        return 2;
    }

    std::vector<u64> factorial(side, 1), inverse_factorial(side, 1);
    for (int i = 1; i <= max_moment; ++i)
        factorial[i] = mod_mul(factorial[i - 1], i, mod);
    inverse_factorial[max_moment] = mod_pow(factorial[max_moment], mod - 2, mod);
    for (int i = max_moment; i > 0; --i)
        inverse_factorial[i - 1] = mod_mul(inverse_factorial[i], i, mod);

    std::vector<u64> dp((mag1_count + 1) * (mag2_count + 1) * area, 0);
    std::vector<u64> even(area), odd(area), convolution(area);
    dp[offset(0, 0, 0, 0)] = 1;

    auto convolve_add = [&](int source, int destination,
                            const std::vector<u64> &positive,
                            const std::vector<u64> &negative) {
        std::fill(even.begin(), even.end(), 0);
        std::fill(odd.begin(), odd.end(), 0);
        std::fill(convolution.begin(), convolution.end(), 0);
        for (int s = 0; s <= max_moment; ++s) {
            for (int r = 0; r <= max_moment; ++r) {
                u64 even_sum = 0;
                u64 odd_sum = 0;
                for (int i = 0; i <= r; i += 2)
                    add_assign(even_sum,
                               mod_mul(dp[source + (r - i) * side + s], positive[i], mod),
                               mod);
                for (int i = 1; i <= r; i += 2)
                    add_assign(odd_sum,
                               mod_mul(dp[source + (r - i) * side + s], positive[i], mod),
                               mod);
                even[r * side + s] = even_sum;
                odd[r * side + s] = odd_sum;
            }
        }
        for (int r = 0; r <= max_moment; ++r) {
            for (int s = 0; s <= max_moment; ++s) {
                u64 sum = 0;
                for (int j = 0; j <= s; j += 2)
                    add_assign(sum,
                               mod_mul(even[r * side + s - j], negative[j], mod), mod);
                for (int j = 1; j <= s; j += 2)
                    add_assign(sum,
                               mod_mul(odd[r * side + s - j], negative[j], mod), mod);
                convolution[r * side + s] = sum;
            }
        }
        for (int index = 0; index < area; ++index)
            add_assign(dp[destination + index], convolution[index], mod);
    };

    const auto started = std::chrono::steady_clock::now();
    for (int position = 0; position < dimension; ++position) {
        const u64 position_root = mod_pow(root, position, mod);
        const u64 position_root_inverse = mod_pow(position_root, mod - 2, mod);
        std::vector<u64> positive1(side), negative1(side), positive2(side), negative2(side);
        positive1[0] = negative1[0] = positive2[0] = negative2[0] = 1;
        for (int i = 1; i <= max_moment; ++i) {
            positive1[i] = mod_mul(positive1[i - 1], position_root, mod);
            negative1[i] = mod_mul(negative1[i - 1], position_root_inverse, mod);
            positive2[i] = mod_mul(positive2[i - 1], mod_mul(position_root, 2, mod), mod);
            negative2[i] = mod_mul(negative2[i - 1], mod_mul(position_root_inverse, 2, mod), mod);
        }
        for (int i = 0; i <= max_moment; ++i) {
            positive1[i] = mod_mul(positive1[i], inverse_factorial[i], mod);
            negative1[i] = mod_mul(negative1[i], inverse_factorial[i], mod);
            positive2[i] = mod_mul(positive2[i], inverse_factorial[i], mod);
            negative2[i] = mod_mul(negative2[i], inverse_factorial[i], mod);
        }

        const int max_a = std::min(mag1_count, position + 1);
        for (int a = max_a; a >= 0; --a) {
            const int max_b = std::min(mag2_count, position + 1 - a);
            for (int b = max_b; b >= 0; --b) {
                if (a + b == 0 || a + b > position + 1) continue;
                const int destination = offset(a, b, 0, 0);
                if (a > 0)
                    convolve_add(offset(a - 1, b, 0, 0), destination, positive1, negative1);
                if (b > 0)
                    convolve_add(offset(a, b - 1, 0, 0), destination, positive2, negative2);
            }
        }
    }

    std::cout << "prime " << mod << " root " << root << "\n";
    for (int moment = 0; moment <= max_moment; ++moment) {
        u64 residue = dp[offset(mag1_count, mag2_count, moment, moment)];
        residue = mod_mul(residue, factorial[moment], mod);
        residue = mod_mul(residue, factorial[moment], mod);
        std::cout << moment << " " << residue << "\n";
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cerr << "elapsed_seconds " << elapsed << "\n";
}
