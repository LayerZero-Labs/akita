#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using u128 = unsigned __int128;

static u64 mod_mul(u64 a, u64 b, u64 p) {
    return static_cast<u64>((static_cast<u128>(a) * b) % p);
}

static u64 mod_pow(u64 a, u64 e, u64 p) {
    u64 r = 1;
    while (e != 0) {
        if (e & 1) r = mod_mul(r, a, p);
        a = mod_mul(a, a, p);
        e >>= 1;
    }
    return r;
}

static inline void add_assign(u64 &x, u64 y, u64 p) {
    u64 z = x + y;
    if (z >= p || z < x) z -= p;
    x = z;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "usage: moments_mod PRIME PRIMITIVE_ROOT MAX_MOMENT [DIMENSION WEIGHT]\n";
        return 2;
    }
    const u64 mod = std::stoull(argv[1]);
    const u64 primitive = std::stoull(argv[2]);
    const int n = std::stoi(argv[3]);
    const int d = argc == 6 ? std::stoi(argv[4]) : 128;
    const int w = argc == 6 ? std::stoi(argv[5]) : 31;
    if (d <= 0 || w < 0 || w > d || (mod - 1) % (2 * d) != 0) {
        std::cerr << "invalid dimension, weight, or modulus\n";
        return 2;
    }
    const int side = n + 1;
    const int area = side * side;
    auto at = [side, area](int k, int a, int b) {
        return k * area + a * side + b;
    };

    const u64 root = mod_pow(primitive, (mod - 1) / (2 * d), mod);
    if (mod_pow(root, 2 * d, mod) != 1 || mod_pow(root, d, mod) != mod - 1) {
        std::cerr << "supplied generator did not produce a primitive 256th root\n";
        return 2;
    }

    std::vector<u64> fact(side, 1), invfact(side, 1);
    for (int i = 1; i <= n; ++i) fact[i] = mod_mul(fact[i - 1], i, mod);
    invfact[n] = mod_pow(fact[n], mod - 2, mod);
    for (int i = n; i > 0; --i) invfact[i - 1] = mod_mul(invfact[i], i, mod);

    std::vector<u64> dp((w + 1) * area, 0);
    std::vector<u64> even(area), odd(area), conv(area);
    dp[at(0, 0, 0)] = 1;

    const auto started = std::chrono::steady_clock::now();
    for (int pos = 0; pos < d; ++pos) {
        const u64 rp = mod_pow(root, pos, mod);
        const u64 rn = mod_pow(rp, mod - 2, mod);
        std::vector<u64> poscoef(side), negcoef(side);
        poscoef[0] = negcoef[0] = 1;
        for (int i = 1; i <= n; ++i) {
            poscoef[i] = mod_mul(mod_mul(poscoef[i - 1], rp, mod), i == 1 ? 1 : 1, mod);
            negcoef[i] = mod_mul(negcoef[i - 1], rn, mod);
        }
        for (int i = 0; i <= n; ++i) {
            poscoef[i] = mod_mul(poscoef[i], invfact[i], mod);
            negcoef[i] = mod_mul(negcoef[i], invfact[i], mod);
        }

        for (int k = std::min(w, pos + 1); k >= 1; --k) {
            std::fill(even.begin(), even.end(), 0);
            std::fill(odd.begin(), odd.end(), 0);
            std::fill(conv.begin(), conv.end(), 0);
            const int prev = (k - 1) * area;

            for (int q = 0; q <= n; ++q) {
                for (int p = 0; p <= n; ++p) {
                    u64 se = 0, so = 0;
                    for (int i = 0; i <= p; i += 2)
                        add_assign(se, mod_mul(dp[prev + (p - i) * side + q], poscoef[i], mod), mod);
                    for (int i = 1; i <= p; i += 2)
                        add_assign(so, mod_mul(dp[prev + (p - i) * side + q], poscoef[i], mod), mod);
                    even[p * side + q] = se;
                    odd[p * side + q] = so;
                }
            }
            for (int p = 0; p <= n; ++p) {
                for (int q = 0; q <= n; ++q) {
                    u64 s = 0;
                    for (int j = 0; j <= q; j += 2)
                        add_assign(s, mod_mul(even[p * side + q - j], negcoef[j], mod), mod);
                    for (int j = 1; j <= q; j += 2)
                        add_assign(s, mod_mul(odd[p * side + q - j], negcoef[j], mod), mod);
                    conv[p * side + q] = s;
                }
            }
            const int dst = k * area;
            for (int idx = 0; idx < area; ++idx) add_assign(dp[dst + idx], conv[idx], mod);
        }
    }

    std::cout << "prime " << mod << " root " << root << "\n";
    for (int m = 0; m <= n; ++m) {
        u64 residue = dp[at(w, m, m)];
        residue = mod_mul(residue, fact[m], mod);
        residue = mod_mul(residue, fact[m], mod);
        std::cout << m << " " << residue << "\n";
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "elapsed_seconds " << elapsed << "\n";
}
