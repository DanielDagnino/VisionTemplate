#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <chrono>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: inference <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  module.eval();

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({1, 3, 224, 224}, torch::requires_grad(false)));

  // Execute the model and turn its output into a tensor.
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  at::Tensor output = module.forward(inputs).toTensor();

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;

//  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/20) << '\n';
  std::cout << output << '\n';

  std::cout << "ok\n";
}
