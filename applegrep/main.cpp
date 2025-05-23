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

// Metal Shader for re matching
// ... I just let LLM to implement the Boyer-Moore-Horspool algorithm
const char* grepShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

constant int ALPHABET_SIZE = 256; // Assuming ASCII characters

kernel void grep_kernel(
    device const char* text [[buffer(0)]],
    device const char* pattern [[buffer(1)]],
    device int* match_positions [[buffer(2)]],  // Buffer to store match positions
    device atomic_int* match_count [[buffer(3)]], // Atomic counter
    uint tid [[thread_position_in_grid]])
{
    // Calculate text length
    uint text_length = 0;
    while (text[text_length] != '\0') text_length++;
    
    // Calculate pattern length
    uint pattern_length = 0;
    while (pattern[pattern_length] != '\0') pattern_length++;
    
    // If pattern is empty or longer than remaining text, return
    if (pattern_length == 0 || tid > text_length - pattern_length) return;
    
    // Preprocess bad-character shift table
    int bad_char_shift[ALPHABET_SIZE];
    
    // Initialize all shifts to pattern length (default shift)
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        bad_char_shift[i] = pattern_length;
    }
    
    // Set shift for characters in pattern (except last)
    for (uint i = 0; i < pattern_length - 1; ++i) {
        bad_char_shift[pattern[i]] = pattern_length - 1 - i;
    }
    
    // BMH search - each thread handles one potential starting position
    int j = pattern_length - 1;
    
    // Compare from right to left
    while (j >= 0 && pattern[j] == text[tid + j]) {
        j--;
    }
    
    if (j < 0) {
        // Pattern found - use atomic operation to ensure unique position
        int count = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        if (count < 1000) {  // Prevent buffer overflow
            match_positions[count] = tid;
        }
    }
}
)";

// Read file
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot read file" << filename << std::endl;
        return "";
    }
    
    return std::string((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
}

int main(int argc, const char* argv[]) {
    std::string text;
    std::string filename;

    if (argc == 2) {
        // Read from stdin
        filename = "stdin";
        text = std::string((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());
    } else if (argc == 3){
        // Read from file
        filename = argv[2];
        text = readFile(filename);
    } else {
        std::cerr << "Usage: " << argv[0] << " <pattern> [file]" << std::endl;
        return 1;
    }
    
    const std::string pattern = argv[1];
    const size_t max_matches = 10;  // Adjust based on expected matches
    
    if (text.empty() || pattern.empty()) {
        std::cout << "Found 0 matches for '" << pattern
                  << "' in file '" << filename << "'" << std::endl;
        return 0;
    }

    // 1. Initialize Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    
    // 2. Complile shader
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(
        NS::String::string(grepShaderSource, NS::UTF8StringEncoding), nullptr, &error);
    
    if (!library) {
        std::cerr << "Failed to compile shader: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }
    
    MTL::Function* grepFunction = library->newFunction(NS::String::string("grep_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(grepFunction, &error);
    
    // 3. Prepare data
    std::vector<char> textData(text.begin(), text.end());
    textData.push_back('\0');
    std::vector<char> patternData(pattern.begin(), pattern.end());
    patternData.push_back('\0');
    
    // 4. Create buffers
    int initialMatchCount = 0;
    std::vector<int> matchPositions(max_matches, 0);
    
    MTL::Buffer* textBuffer = device->newBuffer(textData.data(), textData.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* patternBuffer = device->newBuffer(patternData.data(), patternData.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchCountBuffer = device->newBuffer(&initialMatchCount, sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchPositionsBuffer = device->newBuffer(matchPositions.data(), max_matches * sizeof(int), MTL::ResourceStorageModeShared);

    // 5. Create command queue and buffer
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    // 6. Encode compute command
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(textBuffer, 0, 0);       // buffer 0: text
    computeEncoder->setBuffer(patternBuffer, 0, 1);    // buffer 1: pattern
    computeEncoder->setBuffer(matchPositionsBuffer, 0, 2); // buffer 2: match positions
    computeEncoder->setBuffer(matchCountBuffer, 0, 3); // buffer 3: match count
    
    // 7. Configure threads
    MTL::Size gridSize = MTL::Size(text.size() - pattern.size() + 1, 1, 1);
    NS::UInteger maxThreads = pipelineState->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size(std::min(maxThreads, (NS::UInteger)gridSize.width), 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    
    // 8. Commit and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // 9. Get results (copy data back from GPU)
    int matchCount = *(static_cast<int*>(matchCountBuffer->contents()));
    if (matchCount > max_matches) {
        std::cerr << "Warning: Found " << matchCount << " matches but only "
                  << max_matches << " can be stored" << std::endl;
        matchCount = max_matches;
    }
    memcpy(matchPositions.data(), matchPositionsBuffer->contents(), matchCount * sizeof(int));
    
    std::cout << "Found " << matchCount << " matches for '" << pattern
              << "' in file '" << filename << "'" << std::endl;
    
    // 10. Print matching lines
    std::vector<size_t> line_starts;
    line_starts.push_back(0);
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            line_starts.push_back(i + 1);
        }
    }
    
    for (int i = 0; i < matchCount; ++i) {  // ONLY PROCESS ACTUAL MATCHES
        int pos = matchPositions[i];
        // Find which line contains this match
        size_t line_idx = 0;
        for (; line_idx < line_starts.size() - 1; ++line_idx) {
            if (pos >= line_starts[line_idx] && pos < line_starts[line_idx + 1]) {
                break;
            }
        }
        
        // Extract the line
        size_t line_start = line_starts[line_idx];
        size_t line_end = (line_idx < line_starts.size() - 1)
                         ? line_starts[line_idx + 1] - 1
                         : text.size();
        std::string matching_line = text.substr(line_start, line_end - line_start);
        
        // Print grep-style output
        std::cout << filename << ":" << (line_idx + 1) << ":\t" << matching_line << "\n";
    }
    
    // 10. Free resources
    computeEncoder->release();
    commandBuffer->release();
    commandQueue->release();
    textBuffer->release();
    patternBuffer->release();
    pipelineState->release();
    grepFunction->release();
    library->release();
    device->release();
    
    return 0;
}
