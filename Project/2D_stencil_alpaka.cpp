#include <alpaka/alpaka.hpp>
#include "config.hpp"
#include "workdivision.hpp"

constexpr std::size_t DSIZE = 512;
#define RADIUS 3

void check_matrix_mult(alpaka::BufCpu<int, Dim1D, uint32_t> data1, alpaka::BufCpu<int, Dim1D, uint32_t> data2, alpaka::BufCpu<int, Dim1D, uint32_t> data3) {
    for (auto i = 0; i < DSIZE + 2 * RADIUS; i++) {
        for (auto j = 0; j < DSIZE + 2 * RADIUS; j++) {
            int sum = 0;
            for (auto k = 0; k < DSIZE + 2 * RADIUS; k++) {
                sum += data1[i * (DSIZE + 2 * RADIUS) + k] * data2[k * (DSIZE + 2 * RADIUS) + j];
            }
            if (data3[i * (DSIZE + 2 * RADIUS) + j] != sum) {
                std::cout << "Error at " << i << " " << j << " in matrix mult, got " << data3[i * (DSIZE + 2 * RADIUS) + j] << " expected " << sum << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

void check_stencil(alpaka::BufCpu<int, Dim1D, uint32_t> data1, alpaka::BufCpu<int, Dim1D, uint32_t> data2) {
    for (auto i = 0; i < DSIZE + 2 * RADIUS; i++) {
        for (auto j = 0; j < DSIZE + 2 * RADIUS; j++) {
            if (i < RADIUS || i >= (DSIZE + 2 * RADIUS) - RADIUS || j < RADIUS || j >= (DSIZE + 2 * RADIUS) - RADIUS) {
                if (data1[i * (DSIZE + 2 * RADIUS) + j] != data2[i * (DSIZE + 2 * RADIUS) + j]) {
                    std::cout << "Error at " << i << " " << j << " in stencil, got " << data2[i * (DSIZE + 2 * RADIUS) + j] << " expected " << data1[i * (DSIZE + 2 * RADIUS) + j] << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                int sum = 0;
                for (auto k = -RADIUS; k <= RADIUS; k++) {
                    sum += data1[(i + k) * (DSIZE + 2 * RADIUS) + j];
                    sum += data1[i * (DSIZE + 2 * RADIUS) + j + k];
                }
                sum -= data1[i * (DSIZE + 2 * RADIUS) + j];
                if (data2[i * (DSIZE + 2 * RADIUS) + j] != sum) {
                    std::cout << "Error at " << i << " " << j << " in applying stencil, got " << data2[i * (DSIZE + 2 * RADIUS) + j] << " expected " << sum << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
}

struct stencil_2D_alpaka {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data1, T* __restrict__ data2) const {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadElementIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto const idx = threadElementIdx[0];
        auto const x = idx % (DSIZE + 2 * RADIUS);
        auto const y = idx / (DSIZE + 2 * RADIUS);

        if (x < RADIUS || x >= (DSIZE + 2 * RADIUS) - RADIUS || y < RADIUS || y >= (DSIZE + 2 * RADIUS) - RADIUS) {
            data2[y * (DSIZE + 2 * RADIUS) + x] = data1[y * (DSIZE + 2 * RADIUS) + x];
            return;
        }

        int sum = 0;
        for (auto i = -RADIUS; i <= RADIUS; i++) {
            sum += data1[(y + i) * (DSIZE + 2 * RADIUS) + x];
            sum += data1[y * (DSIZE + 2 * RADIUS) + x + i];
        }
        sum -= data1[y * (DSIZE + 2 * RADIUS) + x];
        data2[y * (DSIZE + 2 * RADIUS) + x] = sum;
    }
};

struct matrix_mult {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data1, T const* __restrict__ data2, T* __restrict__ data3) const {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadElementIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto const idx = threadElementIdx[0];
        auto const x = idx % (DSIZE + 2 * RADIUS);
        auto const y = idx / (DSIZE + 2 * RADIUS);

        int sum = 0;
        for (auto i = 0; i < DSIZE + 2 * RADIUS; i++) {
            sum += data1[y * (DSIZE + 2 * RADIUS) + i] * data2[i * (DSIZE + 2 * RADIUS) + x];
        }
        data3[y * (DSIZE + 2 * RADIUS) + x] = sum;

    }
};

int main(){
    // require at least one device
    std::size_t n = alpaka::getDevCount<Platform>();
    if (n == 0) {
        exit(EXIT_FAILURE);
    }

    //grab single host and first device
    Host host = alpaka::getDevByIdx<HostPlatform>(0u);
    Device device = alpaka::getDevByIdx<Platform>(0u);

    //allocate memory
    auto A = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto B = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto C = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto D = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto E = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});

    //set up random numbers
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::uniform_int_distribution<int> dist{0, 5};

    //fill input1/2 with random numbers
    for (auto i = 0; i < (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS); i++) {
        A[i] = dist(rand);
        B[i] = dist(rand);
        C[i] = A[i];
        D[i] = B[i];
        E[i] = 0;
    }

    //create queue
    auto queue = Queue{device};

    //allocate on device
    auto dev_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto dev_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto dev_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto dev_D = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});
    auto dev_E = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{(DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS)});

    //copy to device
    alpaka::memcpy(queue, dev_A, A);
    alpaka::memcpy(queue, dev_B, B);

    alpaka::memset(queue, dev_C, 0x00);
    alpaka::memset(queue, dev_D, 0x00);
    alpaka::memset(queue, dev_E, 0x00);

    auto div = make_workdiv<Acc1D>(1024, 1024);
    std::cout << "Testing VectorAddKernel with scalar indices with a grid of "
                << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
                << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
                << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
    alpaka::exec<Acc1D>(
        queue, div, stencil_2D_alpaka{}, dev_A.data(), dev_C.data());
    alpaka::exec<Acc1D>(
        queue, div, stencil_2D_alpaka{}, dev_B.data(), dev_D.data());

    alpaka::memcpy(queue, C, dev_C);
    alpaka::memcpy(queue, D, dev_D);
    alpaka::wait(queue);

    check_stencil(A, C);
    check_stencil(B, D);

    //Now we can do matrix multiplication
    alpaka::exec<Acc1D>(
        queue, div, matrix_mult{}, dev_C.data(), dev_D.data(), dev_E.data());

    alpaka::memcpy(queue, E, dev_E);

    alpaka::wait(queue);

    check_matrix_mult(C, D, E);

}