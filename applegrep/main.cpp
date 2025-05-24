#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

const char* grepShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

constant int ALPHABET_SIZE = 256;
constant int STRIDE = 4; // Each thread checks 4 positions

#include <metal_stdlib>
using namespace metal;

kernel void grep_kernel(
    device const char* text [[buffer(0)]],
    device const char* pattern [[buffer(1)]],
    device int* match_positions [[buffer(2)]],
    device atomic_int* match_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]])
{
    uint text_length = 0;
    while (text[text_length] != '\0') text_length++;
    
    uint pattern_length = 0;
    while (pattern[pattern_length] != '\0') pattern_length++;
    
    if (pattern_length == 0 || text_length < pattern_length) return;
    
    uint pos = tid;
    if (pos <= text_length - pattern_length) {
        bool matched = true;
        
        for (uint i = 0; i < pattern_length; ++i) {
            if (text[pos + i] != pattern[i]) {
                matched = false;
                break;
            }
        }
        
        if (matched) {
            int count = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            if (count < 10000) {  // avoid overflow
                match_positions[count] = pos;
            }
        }
    }
}
)";

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot read file: " << filename << std::endl;
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
}

std::vector<int> precomputeBadCharShift(const std::string& pattern) {
    const int ALPHABET_SIZE = 256;
    std::vector<int> bad_char_shift(ALPHABET_SIZE, pattern.length());
    for (size_t i = 0; i < pattern.length() - 1; ++i) {
        bad_char_shift[static_cast<unsigned char>(pattern[i])] = pattern.length() - 1 - i;
    }
    return bad_char_shift;
}

int main(int argc, const char* argv[]) {
    // 1. Input handling
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <pattern> [file]" << std::endl;
        return 1;
    }

    std::string text = (argc == 2) ? std::string(std::istreambuf_iterator<char>(std::cin),
                                   std::istreambuf_iterator<char>())
                                  : readFile(argv[2]);
    const std::string filename = (argc == 2) ? "stdin" : argv[2];
    const std::string pattern = argv[1];
    const size_t max_matches = 10000;

    if (text.empty() || pattern.empty()) {
        std::cout << "Found 0 matches for '" << pattern << "' in '" << filename << "'" << std::endl;
        return 0;
    }

    // 2. Precompute bad_char_shift on CPU
    // auto badCharShift = precomputeBadCharShift(pattern);

    // 3. Metal setup
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    NS::Error* error = nullptr;
    
    // 4. Compile shader
    MTL::Library* library = device->newLibrary(
        NS::String::string(grepShaderSource, NS::UTF8StringEncoding), nullptr, &error);
    if (!library) {
        std::cerr << "Shader compile error: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    MTL::Function* grepFunction = library->newFunction(NS::String::string("grep_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(grepFunction, &error);
    if (!pipelineState) {
        std::cerr << "Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    // 5. Prepare buffers
    std::vector<char> textData(text.begin(), text.end());
    textData.push_back('\0');
    std::vector<char> patternData(pattern.begin(), pattern.end());
    patternData.push_back('\0');
    
    int initialMatchCount = 0;
    std::vector<int> matchPositions(max_matches, 0);

    MTL::Buffer* textBuffer = device->newBuffer(textData.data(), textData.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* patternBuffer = device->newBuffer(patternData.data(), patternData.size(), MTL::ResourceStorageModeShared);
    //MTL::Buffer* badCharShiftBuffer = device->newBuffer(badCharShift.data(), badCharShift.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchCountBuffer = device->newBuffer(&initialMatchCount, sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchPositionsBuffer = device->newBuffer(matchPositions.data(), max_matches * sizeof(int), MTL::ResourceStorageModeShared);

    // 6. Configure dispatch
    const uint stride = 4; // Must match STRIDE in shader
    const uint totalPositions = text.size() - pattern.size() + 1;
    const uint threadCount = (totalPositions + stride - 1) / stride;
    std::cout<<threadCount<<std::endl;
    
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(textBuffer, 0, 0);
    computeEncoder->setBuffer(patternBuffer, 0, 1);
    //computeEncoder->setBuffer(badCharShiftBuffer, 0, 2);
    computeEncoder->setBuffer(matchPositionsBuffer, 0, 3);
    computeEncoder->setBuffer(matchCountBuffer, 0, 4);
    
    // Use 256 threads per group (optimized for most GPUs)
    MTL::Size gridSize = MTL::Size(threadCount, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(1, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // 7. Process results
    int matchCount = *(static_cast<int*>(matchCountBuffer->contents()));
    if (matchCount > max_matches) {
        std::cerr << "Warning: Truncated " << matchCount << " matches to " << max_matches << std::endl;
        matchCount = max_matches;
    }
    memcpy(matchPositions.data(), matchPositionsBuffer->contents(), matchCount * sizeof(int));

    // 8. Output results
    std::cout << "Found " << matchCount << " matches for '" << pattern
              << "' in '" << filename << "'" << std::endl;
    
    // Build line index
    std::vector<size_t> lineStarts = {0};
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') lineStarts.push_back(i + 1);
    }
    
    for (int i = 0; i < matchCount; ++i) {
        size_t pos = matchPositions[i];
        auto it = std::upper_bound(lineStarts.begin(), lineStarts.end(), pos);
        size_t lineIdx = std::distance(lineStarts.begin(), it) - 1;
        
        size_t lineStart = lineStarts[lineIdx];
        size_t lineEnd = (lineIdx + 1 < lineStarts.size()) ? lineStarts[lineIdx + 1] - 1 : text.size();
        
        std::string line = text.substr(lineStart, lineEnd - lineStart);
        std::cout << filename << ":" << (lineIdx + 1) << ":\t" << line << "\n";
    }

    // 9. Cleanup
    textBuffer->release();
    patternBuffer->release();
    //badCharShiftBuffer->release();
    matchCountBuffer->release();
    matchPositionsBuffer->release();
    pipelineState->release();
    grepFunction->release();
    library->release();
    commandQueue->release();
    device->release();

    return 0;
}
