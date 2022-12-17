#include <iostream>
#include <Eigen/Dense>
#include <numbers>
#include <filesystem>
#include <fstream>
#include <json/json.h>
#include <chrono>

#ifdef IS_FLOAT
#define MATRIX_TYPE Eigen::MatrixXf
#define VECTOR_TYPE Eigen::VectorXf
#define TYPE float
#else
#define MATRIX_TYPE Eigen::MatrixXd
#define VECTOR_TYPE Eigen::VectorXd
typedef long double TYPE;
#endif


#define MCD 1
#define NEWMARK 2


// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
TYPE get_dt(const MATRIX_TYPE &masses, const MATRIX_TYPE &stiffness) {
    const TYPE eigen_max = (masses.inverse() * stiffness).eigenvalues().real().maxCoeff();
    const TYPE omega_max = sqrt(eigen_max);
    const TYPE min_period = TYPE(2) * std::numbers::pi_v<TYPE> / omega_max;
    return min_period / TYPE(10);
}

// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
void mechanics_mcd(const MATRIX_TYPE &masses, const MATRIX_TYPE &demp, const MATRIX_TYPE &stiffness, TYPE dt,
                   VECTOR_TYPE &V_0, VECTOR_TYPE &V_1, const std::function<const VECTOR_TYPE(const size_t)> &F,
                   const size_t N,
                   const std::function<void(VECTOR_TYPE &)> &boundaries = nullptr,
                   const std::function<void(const VECTOR_TYPE &, const size_t)> &callback = nullptr) {
    const TYPE sqr_dt = dt * dt;
    const TYPE double_dt = TYPE(2) * dt;
    const MATRIX_TYPE M_div_sqr_dt = masses / sqr_dt;
    const MATRIX_TYPE D_div_double_dt = demp / double_dt;
    const MATRIX_TYPE Q_1 = (M_div_sqr_dt + D_div_double_dt).inverse();
    const MATRIX_TYPE Q_2 = Q_1 * (2 * M_div_sqr_dt - stiffness);
    const MATRIX_TYPE Q_3 = Q_1 * (M_div_sqr_dt - D_div_double_dt);
    for (size_t idx = 0; idx < N; idx++) {
        VECTOR_TYPE V_next = Q_1 * F(idx) + Q_2 * V_1 - Q_3 * V_0;
        if (boundaries != nullptr) {
            boundaries(V_next);
        }
        if (callback != nullptr) {
            callback(V_next, idx);
        }
        V_0 = V_1;
        V_1 = V_next;
    }
}

// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
void mechanics_mcd(const MATRIX_TYPE &masses, const MATRIX_TYPE &demp, const MATRIX_TYPE &stiffness,
                   VECTOR_TYPE &V_0, VECTOR_TYPE &V_1, const std::function<const VECTOR_TYPE(const size_t)> &F,
                   const size_t N,
                   const std::function<void(VECTOR_TYPE &)> &boundaries = nullptr,
                   const std::function<void(const VECTOR_TYPE &, const size_t)> &callback = nullptr) {
    mechanics_mcd(masses, demp, stiffness,
                  get_dt(masses, stiffness),
                  V_0, V_1, F, N, boundaries, callback);
}

// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
void mechanics_newmark(const MATRIX_TYPE &masses, const MATRIX_TYPE &demp, const MATRIX_TYPE &stiffness,
                       VECTOR_TYPE &acceleration, VECTOR_TYPE &speed, VECTOR_TYPE &v, TYPE dt,
                       const std::function<const VECTOR_TYPE(const size_t)> &F, const size_t N,
                       const std::function<void(VECTOR_TYPE &, VECTOR_TYPE &, VECTOR_TYPE &)> &boundaries = nullptr,
                       const std::function<void(const VECTOR_TYPE &, const VECTOR_TYPE &, const VECTOR_TYPE &,
                                                const size_t)> &callback = nullptr) {
    const MATRIX_TYPE Q_1 = TYPE(2) * masses / dt;
    const MATRIX_TYPE Q_2 = (Q_1 + demp) / dt;
    const MATRIX_TYPE inv_Z = (Q_2 + stiffness).inverse();
    const TYPE double_div_sqr_dt = TYPE(2) / (dt * dt);
    for (size_t idx = 0; idx < N; idx++) {
        const VECTOR_TYPE R_i_next = F(idx + 1) + masses * acceleration + Q_1 * speed + Q_2 * v;
        VECTOR_TYPE V_next = R_i_next * inv_Z;
        if (boundaries != nullptr) {
            boundaries(V_next, speed, acceleration);
        }
        const VECTOR_TYPE acceleration_next = double_div_sqr_dt * (V_next - v - speed * dt) - acceleration;
        speed = speed + ((acceleration + acceleration_next) * dt) / TYPE(2);
        acceleration = acceleration_next;
        v = V_next;
        if (callback != nullptr) {
            callback(v, acceleration, speed, idx + 1);
        }
    }
}

void mechanics_newmark(const MATRIX_TYPE &masses, const MATRIX_TYPE &demp, const MATRIX_TYPE &stiffness,
                       VECTOR_TYPE &acceleration, VECTOR_TYPE &speed, VECTOR_TYPE &v,
                       const std::function<const VECTOR_TYPE(const size_t)> &F, const size_t N,
                       const std::function<void(VECTOR_TYPE &, VECTOR_TYPE &, VECTOR_TYPE &)> &boundaries = nullptr,
                       const std::function<void(const VECTOR_TYPE &, const VECTOR_TYPE &, const VECTOR_TYPE &,
                                                const size_t)> &callback = nullptr) {
    mechanics_newmark(
            masses, demp, stiffness,
            acceleration, speed, v, get_dt(masses, stiffness),
            F, N,
            boundaries,
            callback
    );
}


MATRIX_TYPE masses_matrix(const TYPE A, const TYPE J, const TYPE L, const TYPE rho) {
    const auto total_modifier = rho * A * L / ((TYPE) 420);
    const auto four_sqr_l = 4 * L * L;
    const auto one_hundred_fourty_j_div_a = 140 * J / A;
    const auto twenty_two_l = 22 * L;
    const auto thirteen_l = 13 * L;
    const auto minus_three_l_mul_l = -3 * L * L;
    const auto seventy_j_div_a = 70 * J / A;
    MATRIX_TYPE masses(12, 12);
    for (unsigned short i = 0; i < 12; i++) {
        TYPE val = -1;
        switch (i) {
            case 0:
            case 6:
                val = 140;
                break;
            case 1:
            case 2:
            case 7:
            case 8:
                val = 156;
                break;
            case 3:
            case 9:
                val = one_hundred_fourty_j_div_a;
                break;
            case 4:
            case 5:
            case 10:
            case 11:
                val = four_sqr_l;
                break;
            default:
                exit(2);
        }
        masses.coeffRef(i, i) = val;
        for (unsigned short j = 0; j < i; j++) {
            masses.coeffRef(i, j) = 0;
            masses.coeffRef(j, i) = 0;
            if (i == 4 && j == 2) {
                masses.coeffRef(i, j) = -twenty_two_l;
                masses.coeffRef(j, i) = -twenty_two_l;
            }
            if (i == 5 && j == 1) {
                masses.coeffRef(i, j) = twenty_two_l;
                masses.coeffRef(j, i) = twenty_two_l;
            }
            if (i == 6 && j == 0) {
                masses.coeffRef(i, j) = 70;
                masses.coeffRef(j, i) = 70;
            }
            if (i == 7) {
                if (j == 1) {
                    masses.coeffRef(i, j) = 54;
                    masses.coeffRef(j, i) = 54;
                }
                if (j == 5) {
                    masses.coeffRef(i, j) = thirteen_l;
                    masses.coeffRef(j, i) = thirteen_l;
                }
            }
            if (i == 8) {
                if (j == 2) {
                    masses.coeffRef(i, j) = 54;
                    masses.coeffRef(j, i) = 54;
                }
                if (j == 4) {
                    masses.coeffRef(i, j) = -thirteen_l;
                    masses.coeffRef(j, i) = -thirteen_l;
                }
            }
            if (i == 9 && j == 3) {
                masses.coeffRef(i, j) = seventy_j_div_a;
                masses.coeffRef(j, i) = seventy_j_div_a;
            }
            if (i == 10) {
                if (j == 2) {
                    masses.coeffRef(i, j) = thirteen_l;
                    masses.coeffRef(j, i) = thirteen_l;
                }
                if (j == 4) {
                    masses.coeffRef(i, j) = minus_three_l_mul_l;
                    masses.coeffRef(j, i) = minus_three_l_mul_l;
                }
                if (j == 8) {
                    masses.coeffRef(i, j) = twenty_two_l;
                    masses.coeffRef(j, i) = twenty_two_l;
                }
            }
            if (i == 11) {
                if (j == 1) {
                    masses.coeffRef(i, j) = -thirteen_l;
                    masses.coeffRef(j, i) = -thirteen_l;
                }
                if (j == 5) {
                    masses.coeffRef(i, j) = minus_three_l_mul_l;
                    masses.coeffRef(j, i) = minus_three_l_mul_l;
                }
                if (j == 7) {
                    masses.coeffRef(i, j) = -twenty_two_l;
                    masses.coeffRef(j, i) = -twenty_two_l;
                }
            }
        }
    }
    return total_modifier * masses;
}


MATRIX_TYPE stiffness_matrix(const TYPE E, const TYPE A, const TYPE L, const TYPE I_z, const TYPE I_y, const TYPE G,
                             const TYPE J) {
    const auto l_pow_two = L * L;
    const auto l_pow_three = l_pow_two * L;
    const auto c0 = E * A / L;
    const auto c1 = 12 * E * I_y / l_pow_three;
    const auto c2 = 12 * E * I_z / l_pow_three;
    const auto c3 = G * J / L;
    const auto c4 = 4 * E * I_y / L;
    const auto c5 = 4 * E * I_z / L;
    const auto c6 = 6 * E * I_y / l_pow_two;
    const auto c7 = 6 * E * I_z / l_pow_two;
    const auto c8 = c2 / 2 * L;
    const auto c9 = c1 / 2 * L;
    const auto c10 = c4 / 2;
    const auto c11 = c5 / 2;
    MATRIX_TYPE stiffness(12, 12);
    for (unsigned short i = 0; i < 12; i++) {
        TYPE val = -1;
        switch (i) {
            case 0:
            case 6:
                val = c0;
                break;
            case 1:
            case 7:
                val = c2;
                break;
            case 2:
            case 8:
                val = c1;
                break;
            case 3:
            case 9:
                val = c3;
                break;
            case 4:
            case 10:
                val = c4;
                break;
            case 5:
            case 11:
                val = c5;
                break;
            default:
                exit(2);
        }
        stiffness.coeffRef(i, i) = val;
        for (unsigned short j = 0; j < i; j++) {
            stiffness.coeffRef(i, j) = 0;
            stiffness.coeffRef(j, i) = 0;
            if (i == 4 && j == 2) {
                stiffness.coeffRef(i, j) = -c6;
                stiffness.coeffRef(j, i) = -c6;
            }
            if (i == 5 && j == 1) {
                stiffness.coeffRef(i, j) = c7;
                stiffness.coeffRef(j, i) = c7;
            }
            if (i == 6 && j == 0) {
                stiffness.coeffRef(i, j) = -c0;
                stiffness.coeffRef(j, i) = -c0;
            }
            if (i == 7) {
                if (j == 1) {
                    stiffness.coeffRef(i, j) = -c2;
                    stiffness.coeffRef(j, i) = -c2;
                }
                if (j == 5) {
                    stiffness.coeffRef(i, j) = -c8;
                    stiffness.coeffRef(j, i) = -c8;
                }
            }
            if (i == 8) {
                if (j == 2) {
                    stiffness.coeffRef(i, j) = -c1;
                    stiffness.coeffRef(j, i) = -c1;
                }
                if (j == 4) {
                    stiffness.coeffRef(i, j) = c9;
                    stiffness.coeffRef(j, i) = c9;
                }
            }
            if (i == 9 && j == 3) {
                stiffness.coeffRef(i, j) = -c3;
                stiffness.coeffRef(j, i) = -c3;
            }
            if (i == 10) {
                if (j == 2) {
                    stiffness.coeffRef(i, j) = -c9;
                    stiffness.coeffRef(j, i) = -c9;
                }
                if (j == 4) {
                    stiffness.coeffRef(i, j) = c10;
                    stiffness.coeffRef(j, i) = c10;
                }
                if (j == 8) {
                    stiffness.coeffRef(i, j) = c9;
                    stiffness.coeffRef(j, i) = c9;
                }
            }
            if (i == 11) {
                if (j == 1) {
                    stiffness.coeffRef(i, j) = c8;
                    stiffness.coeffRef(j, i) = c8;
                }
                if (j == 5) {
                    stiffness.coeffRef(i, j) = c11;
                    stiffness.coeffRef(j, i) = c11;
                }
                if (j == 7) {
                    stiffness.coeffRef(i, j) = -c8;
                    stiffness.coeffRef(j, i) = -c8;
                }
            }
        }
    }
    return stiffness;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Invalid usage. run 'mechanics <<json-config-path>>'\n";
        exit(1);
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const auto jsonConfigFilePathString = std::string(argv[1]);
    const auto jsonConfigFilePath = std::filesystem::path(jsonConfigFilePathString);
    if (std::filesystem::exists(jsonConfigFilePath) && std::filesystem::is_regular_file(jsonConfigFilePath)) {
        std::ifstream config(jsonConfigFilePathString, std::ifstream::binary);
        Json::Reader reader;
        Json::Value root;
        reader.parse(config, root);
        const auto method_type = root.get("method", (int) (-1)).as<int>();
        if (method_type == -1) {
            exit(4);
        }
        const auto rho = root.get("rho", (double) (-1)).as<double>();
        if (rho < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto J = root.get("J", (double) (-1)).as<double>();
        if (J < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto A = root.get("A", (double) (-1)).as<double>();
        if (A < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto L = root.get("L", (double) (-1)).as<double>();
        if (L < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto I_y = root.get("I_y", (double) (-1)).as<double>();
        if (I_y < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto I_z = root.get("I_z", (double) (-1)).as<double>();
        if (I_z < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto E = root.get("E", (double) (-1)).as<double>();
        if (E < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto G = root.get("G", (double) (-1)).as<double>();
        if (G < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto alpha = root.get("alpha", (double) (-1)).as<double>();
        if (alpha < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto beta = root.get("beta", (double) (-1)).as<double>();
        if (beta < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto dt = root.get("dt", (double) (-1)).as<double>();
        const auto N = root.get("N", -1).as<long long>();
        if (N < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        const auto f_data_file_path_string = std::string(root.get("f_dat_file_path", "").as<std::string>());
        const auto f_data_file_path = std::filesystem::path(f_data_file_path_string);
        MATRIX_TYPE fs(N, 12);
        if (std::filesystem::exists(f_data_file_path) && std::filesystem::is_regular_file(f_data_file_path)) {
            std::ifstream fs_data(f_data_file_path_string);
            if (fs_data.is_open()) {
                for (size_t i = 0; i < N; i++) {
                    for (unsigned short j = 0; j < 12; j++) {
                        TYPE val;
                        fs_data >> val;
                        fs.coeffRef(i, j) = val;
                    }
                }
                fs_data.close();
            }
        } else {
            std::cout << "f_dat_file_path is wrong!\n";
            exit(3);
        }
        if (config.is_open()) { config.close(); }
        const auto masses_matrix_res = masses_matrix(A, J, L, rho);
        const auto stiffness_matrix_res = stiffness_matrix(E, A, L, I_z, I_y, G, J);
        const auto demp_matrix_res = alpha * masses_matrix_res + beta * stiffness_matrix_res;
//        std::cout << masses_matrix_res << '\n' << stiffness_matrix_res << demp_matrix_res << '\n';
        if (!(std::filesystem::exists("./output") && std::filesystem::is_directory("./output"))) {
            std::filesystem::create_directories("./output");
        }
        if (method_type == MCD) {
            VECTOR_TYPE v_0(12);
            VECTOR_TYPE v_1(12);
            for (unsigned short i = 0; i < 12; i++) {
                v_0.coeffRef(i) = 0;
                v_1.coeffRef(i) = 0;
            }
            std::ofstream output("./output/result.txt");
            const auto mechanics_callback = [&output](const VECTOR_TYPE &v, const size_t idx) {
                const auto s = v.size();
                output << idx << ' ';
                for (size_t k = 0; k < s; k++) {
                    if (k + 1 < s) { output << v.coeff(k) << ' '; }
                    else { output << v.coeff(k); };
                }
                output << '\n';
            };
            const auto boundaries = [](VECTOR_TYPE &v) {
                const auto s = v.size();
                for (size_t k = 0; k < 6; k++) {
                    v.coeffRef(k) = 0;
                }
            };
            if (dt > 0) {
                mechanics_mcd(
                        masses_matrix_res, demp_matrix_res, stiffness_matrix_res,
                        dt, v_0, v_1,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, boundaries, mechanics_callback
                );
            } else {
                mechanics_mcd(
                        masses_matrix_res, demp_matrix_res, stiffness_matrix_res,
                        v_0, v_1,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, boundaries, mechanics_callback
                );
            }
            if (output.is_open()) { output.close(); }
        }
        if (method_type == NEWMARK) {
            VECTOR_TYPE v(12);
            VECTOR_TYPE speed(12);
            VECTOR_TYPE acceleration(12);
            for (unsigned short i = 0; i < 12; i++) {
                v.coeffRef(i) = 0;
                speed.coeffRef(i) = 0;
                acceleration.coeffRef(i) = 0;
            }
            std::ofstream output("./output/result.txt");
            const auto mechanics_callback = [&output](const VECTOR_TYPE &v, const VECTOR_TYPE &acceleration,
                                                      const VECTOR_TYPE &speed, const size_t idx) {
                const auto s = v.size();
                output << idx << '\n';
                for (size_t k = 0; k < s; k++) {
                    const auto v_c = v.coeff(k);
                    if (k + 1 < s) { output << v_c << ' '; }
                    else { output << v_c; };
                }
                output << '\n';
                for (size_t k = 0; k < s; k++) {
                    const auto s_c = speed.coeff(k);
                    if (k + 1 < s) { output << s_c << ' '; }
                    else { output << s_c; };
                }
                output << '\n';
                for (size_t k = 0; k < s; k++) {
                    const auto a_c = acceleration.coeff(k);
                    if (k + 1 < s) { output << a_c << ' '; }
                    else { output << a_c; };
                }
                output << '\n';
            };
            const auto boundaries = [](VECTOR_TYPE &v, VECTOR_TYPE &speed, VECTOR_TYPE &acceleration) {
                const auto s = v.size();
                for (size_t k = 0; k < 6; k++) {
                    v.coeffRef(k) = 0;
                    speed.coeffRef(k) = 0;
                    acceleration.coeffRef(k) = 0;
                }
            };
            if (dt > 0) {
                mechanics_newmark(
                        masses_matrix_res, demp_matrix_res, stiffness_matrix_res,
                        acceleration, speed, v, dt,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, boundaries, mechanics_callback
                );
            } else {
                mechanics_newmark(
                        masses_matrix_res, demp_matrix_res, stiffness_matrix_res,
                        acceleration, speed, v,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, boundaries, mechanics_callback
                );
            }
            if (output.is_open()) {
                output.close();
            }
        }
    } else {
        std::cout << "Configuration file doesn't exist!\n";
        exit(3);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;
    return 0;
}
