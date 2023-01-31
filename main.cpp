#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define EIGEN_VECTORIZE_SSE3
#define EIGEN_VECTORIZE_SSSE3
#define EIGEN_VECTORIZE_SSE4_1
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_FMA
#endif


#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <numbers>
#include <filesystem>
#include <fstream>
#include <json/json.h>
#include <chrono>

#define IS_FLOAT

#ifdef IS_FLOAT
#define VECTOR_TYPE Eigen::VectorXf
typedef float TYPE;
typedef Eigen::SparseMatrix<float> MATRIX_TYPE;
#else
#define VECTOR_TYPE Eigen::VectorXd
typedef long double TYPE;
typedef Eigen::SparseMatrix<double> MATRIX_TYPE;
#endif

#define MCD 1
#define NEWMARK 2


// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
TYPE get_dt(const MATRIX_TYPE &masses, const MATRIX_TYPE &stiffness) {
    const TYPE eigen_max = (masses.toDense().inverse() * stiffness).eigenvalues().real().maxCoeff();
    const TYPE omega_max = sqrt(eigen_max);
    const TYPE min_period = TYPE(2) * std::numbers::pi_v<TYPE> / omega_max;
    return min_period / TYPE(10);
}

// https://vk.com/doc131581930_645459975?hash=grFJfOlaOMeM59eYHGvsUlgjeqmQXIOW7vmJMt7K6ao&dl=gQSajjmLbAK969UEMJv8hx9zAV1QPXXu7KgxsEnH0dg
void mechanics_mcd(const MATRIX_TYPE &masses, const MATRIX_TYPE &demp, const MATRIX_TYPE &stiffness, TYPE dt,
                   VECTOR_TYPE &V_0, VECTOR_TYPE &V_1, const std::function<const VECTOR_TYPE(const size_t)> &F,
                   const size_t N,
                   const std::function<void(const VECTOR_TYPE &, const size_t)> &callback = nullptr) {
    const TYPE sqr_dt = dt * dt;
    const TYPE double_dt = TYPE(2) * dt;
    const MATRIX_TYPE M_div_sqr_dt = masses / sqr_dt;
    const MATRIX_TYPE D_div_double_dt = demp / double_dt;
    const MATRIX_TYPE Q_1 = (M_div_sqr_dt + D_div_double_dt);
    const MATRIX_TYPE Q_2 = (2 * M_div_sqr_dt - stiffness);
    const MATRIX_TYPE Q_3 = (M_div_sqr_dt - D_div_double_dt);
    for (size_t idx = 0; idx < N; idx++) {
    	Eigen::SimplicialLDLT<Eigen::SparseMatrix<TYPE>> solver(Q_1);
		VECTOR_TYPE V_next = solver.solve(F(idx) + Q_2 * V_1 - Q_3 * V_0);
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
                   const std::function<void(const VECTOR_TYPE &, const size_t)> &callback = nullptr) {
    mechanics_mcd(masses, demp, stiffness,
                  get_dt(masses, stiffness),
                  V_0, V_1, F, N, callback);
}

void mechanics_newmark(
			const MATRIX_TYPE& masses, const MATRIX_TYPE& demp, const MATRIX_TYPE& stiffness,
			const TYPE& dt, const TYPE& alpha, const TYPE& delta, const size_t N,
			VECTOR_TYPE& acceleration, VECTOR_TYPE& speed, VECTOR_TYPE& displacement,
			const std::function<const VECTOR_TYPE(const size_t)> &F,
			const std::function<void(const VECTOR_TYPE &, const VECTOR_TYPE &, const VECTOR_TYPE &, const size_t)> &callback = nullptr
) {
	const auto a0 = 1 / (alpha * dt * dt);
	const auto a1 = delta / (alpha * dt);
	const auto a2 = 1 / (alpha * dt);
	const auto a3 = 1 / (2 * alpha) - 1;
	const auto a4 = delta / alpha - 1;
	const auto a5 = (dt / 2) * (delta / alpha - 2);
	const auto matrix = a0 * masses + a1 * demp + stiffness;
	const auto callback_specified = (callback != nullptr);
	std::cout << "Starting Newmark method with parameters:\n"
				<< "\tdt	= " << dt << '\n'
				<< "\talpha	= " << alpha << '\n'
				<< "\tdelta	= " << delta << '\n'
				<< "\tN 	= " << N << '\n'
				<< "\ta0	= " << a0 << '\n'
				<< "\ta1	= " << a1 << '\n'
				<< "\ta2	= " << a2 << '\n'
				<< "\ta3	= " << a3 << '\n'
				<< "\ta4	= " << a4 << '\n'
				<< "\ta5	= " << a5 << '\n';
	size_t i;
	for (i = 0; i < N; i++) {
		const auto F_n_next = F(i);
		const auto masses_multiplied_argument = masses * (a0 * displacement + a2 * speed + a3 * acceleration);
		const auto demp_multiplied_argument = demp * (a1 * displacement + a4 * speed + a5 * acceleration);
		const auto right_part = F_n_next + masses_multiplied_argument + demp_multiplied_argument;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<TYPE>> solver(matrix);
		const auto displacement_next = solver.solve(right_part);
		const auto speed_next = a1 * (displacement_next - displacement) - a4 * (speed) - a5 * acceleration;
		const auto acceleration_next = a0 * (displacement_next - displacement) - a2 * speed - a3 * acceleration;
		if (callback_specified) {
			callback(displacement_next, speed_next, acceleration_next, i);
		}
		acceleration = acceleration_next;
		speed = speed_next;
		displacement = displacement_next;
	}
}


void ensemble(MATRIX_TYPE& appendable_matrix, MATRIX_TYPE& appended_matrix, const size_t start_from_row, const size_t start_from_col) {
    size_t i, j;
    appendable_matrix.conservativeResize(appended_matrix.rows() - start_from_row + appendable_matrix.rows(), appended_matrix.cols() - start_from_col + appendable_matrix.cols());
    for (i = 0; i < appended_matrix.rows(); i++) {
        for (j = 0; j < appended_matrix.cols(); j++) {
            appendable_matrix.coeffRef(i + start_from_row, j + start_from_col) = appended_matrix.coeffRef(i, j);
        }
    }
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


void process_super_element(
		const Json::Value& super_element,
		MATRIX_TYPE*& masses_full,
		MATRIX_TYPE*& demp_full,
		MATRIX_TYPE*& stiffness_full,
		size_t index
) {
	const auto rho = super_element.get("rho", (double) (-1)).as<double>();
    if (rho < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto J = super_element.get("J", (double) (-1)).as<double>();
    if (J < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto A = super_element.get("A", (double) (-1)).as<double>();
    if (A < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto L = super_element.get("L", (double) (-1)).as<double>();
    if (L < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto I_y = super_element.get("I_y", (double) (-1)).as<double>();
    if (I_y < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto I_z = super_element.get("I_z", (double) (-1)).as<double>();
    if (I_z < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto E = super_element.get("E", (double) (-1)).as<double>();
    if (E < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto G = super_element.get("G", (double) (-1)).as<double>();
    if (G < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto alpha = super_element.get("alpha", (double) (-1)).as<double>();
    if (alpha < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    const auto beta = super_element.get("beta", (double) (-1)).as<double>();
    if (beta < 0) {
        std::cout << "Check given params, some of them are less than 0!\n";
        exit(4);
    }
    auto masses_matrix_res = masses_matrix(A, J, L, rho);
    auto stiffness_matrix_res = stiffness_matrix(E, A, L, I_z, I_y, G, J);
    MATRIX_TYPE demp_matrix_res = alpha * masses_matrix_res + beta * stiffness_matrix_res;
    if (masses_full == nullptr) {
    	masses_full = new MATRIX_TYPE(masses_matrix_res);
    	demp_full = new MATRIX_TYPE(demp_matrix_res);
    	stiffness_full = new MATRIX_TYPE(stiffness_matrix_res);
	} else {
		const auto from = (index) * 6;
		ensemble(
			*masses_full,
			masses_matrix_res,
			from,
			from
		);
		ensemble(
			*demp_full,
			demp_matrix_res,
			from,
			from
		);
		ensemble(
			*stiffness_full,
			stiffness_matrix_res,
			from,
			from
		);
	}
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
        
        auto dt = root.get("dt", (double) (-1)).as<double>();
        const auto N = root.get("N", -1).asInt64();
        if (N < 0) {
            std::cout << "Check given params, some of them are less than 0!\n";
            exit(4);
        }
        
        
        const auto f_data_file_path_string = std::string(root.get("f_dat_file_path", "").as<std::string>());
        const auto f_data_file_path = std::filesystem::path(f_data_file_path_string);

        if (!std::filesystem::exists(f_data_file_path)) {
            std::cout << "Configuration file doesn't exist!\n";
            exit(3);
        }
        
        auto super_elements = root.get("elements", Json::Value(-1));
        
        if (super_elements.isInt() && super_elements.asInt() == -1) {
        	std::cout << "Super elements not specified!\n";
        	exit(4);
		}
		
		MATRIX_TYPE* masses_matrix_res = nullptr;
		MATRIX_TYPE* stiffness_matrix_res = nullptr;
		MATRIX_TYPE* demp_matrix_res = nullptr;
		
		if (super_elements.isArray()) {
			int i;
			for (i = 0; i < super_elements.size(); i++) {
				process_super_element(super_elements[i], masses_matrix_res, demp_matrix_res, stiffness_matrix_res, i);
			}
		} else if (super_elements.isObject()) {
			process_super_element(super_elements, masses_matrix_res, demp_matrix_res, stiffness_matrix_res, 0);
		} else {
			std::cout << "Super element('s) must be object or array.\n";
			exit(3);
		}
		
		if (masses_matrix_res == nullptr) {
			std::cout << "Masses was null!\n";
			exit(3);
		}

        const auto nrows = (*stiffness_matrix_res).rows() - 6;
        const auto ncols = (*stiffness_matrix_res).cols() - 6;

        *masses_matrix_res = (*masses_matrix_res).bottomRightCorner(nrows, ncols);
        *stiffness_matrix_res = (*stiffness_matrix_res).bottomRightCorner(nrows, ncols);
        *demp_matrix_res = (*demp_matrix_res).bottomRightCorner(nrows, ncols);

		const size_t initial_conditions_vectors_size = masses_matrix_res->rows();
        
        MATRIX_TYPE fs(N, initial_conditions_vectors_size);
        if (std::filesystem::exists(f_data_file_path) && std::filesystem::is_regular_file(f_data_file_path)) {
            std::ifstream fs_data(f_data_file_path_string);
            if (fs_data.is_open()) {
                for (size_t i = 0; i < N; i++) {
                    for (unsigned short j = 0; j < initial_conditions_vectors_size; j++) {
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

        if (!(std::filesystem::exists("./output") && std::filesystem::is_directory("./output"))) {
            std::filesystem::create_directories("./output");
        }
        
        std::ofstream output("./output/result.txt");
        
        output << "[M] = \n" << *masses_matrix_res << '\n' << "[C] = \n" << *demp_matrix_res << '\n' << "[K] = \n" << *stiffness_matrix_res << '\n';
        
        if (method_type == MCD) {
            VECTOR_TYPE v_0(initial_conditions_vectors_size);
            VECTOR_TYPE v_1(initial_conditions_vectors_size);
            for (unsigned short i = 0; i < initial_conditions_vectors_size; i++) {
                v_0.coeffRef(i) = 0;
                v_1.coeffRef(i) = 0;
            }
            const auto mechanics_callback = [&output](const VECTOR_TYPE &v, const size_t idx) {
                output << idx << '\n';
                output << v;
                output << '\n';
            };
            if (dt > 0) {
                mechanics_mcd(
                        *masses_matrix_res, *demp_matrix_res, *stiffness_matrix_res,
                        dt, v_0, v_1,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, mechanics_callback
                );
            } else {
                mechanics_mcd(
                        *masses_matrix_res, *demp_matrix_res, *stiffness_matrix_res,
                        v_0, v_1,
                        [fs](const size_t i) {
                            return fs.row(i);
                        }, N, mechanics_callback
                );
            }
        }
        if (method_type == NEWMARK) {
        	const auto integration_alpha = root.get("newmark_alpha", (double)(-1)).as<double>();
        	const auto integration_delta = root.get("newmark_delta", (double)(-1)).as<double>();
            VECTOR_TYPE v(initial_conditions_vectors_size);
            VECTOR_TYPE speed(initial_conditions_vectors_size);
            VECTOR_TYPE acceleration(initial_conditions_vectors_size);
            for (unsigned short i = 0; i < initial_conditions_vectors_size; i++) {
                v.coeffRef(i) = 0;
                speed.coeffRef(i) = 0;
                acceleration.coeffRef(i) = 0;
            }
            const auto mechanics_callback = [&output](const VECTOR_TYPE &v, const VECTOR_TYPE &speed, const VECTOR_TYPE &acceleration, const size_t idx) {
                output << idx << '\n';
                output << "displacement\n" << v << '\n';
                output << "speed\n" << speed << '\n';
                output << "acceleration\n" << acceleration << '\n';
            };
            const auto boundaries = nullptr;
            if (dt > 0) {
                mechanics_newmark(
                        *masses_matrix_res, *demp_matrix_res, *stiffness_matrix_res,
                        dt, integration_alpha, integration_delta, N,
                        acceleration, speed, v,
                        [fs](const size_t i) {
                            return fs.row(i);
                        },
                        mechanics_callback
                );
            } else {
            	const auto max_time = root.get("max_time", double(-1)).as<double>();
            	if (max_time < 0) {
            		std::cout << "Check given params, some of them are less than 0!\n";
            		exit(4);
				}
				dt = max_time / N;
                mechanics_newmark(
                        *masses_matrix_res, *demp_matrix_res, *stiffness_matrix_res,
                        dt, integration_alpha, integration_delta, N,
                        acceleration, speed, v,
                        [fs](const size_t i) {
                            return fs.row(i);
                        },
                        mechanics_callback
                );
            }
        }
        if (output.is_open()) {
            output.close();
        }
        if (config.is_open()) {
        	config.close();
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
