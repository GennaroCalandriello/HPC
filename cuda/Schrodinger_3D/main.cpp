#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>

#include "ISFlow.cu"

int main(int argc, char **argv) {
    // Initialize the ISFlow object
    ISFlow isflow;

    // Read the input image
    std::string input_image = "input.jpg";
    isflow.read_image(input_image);

    // Read the mask image
    std::string mask_image = "mask.jpg";
    isflow.read_mask(mask_image);

    // Compute the ISFlow
    isflow.compute_isflow();

    // Write the output image
    std::string output_image = "output.jpg";
    isflow.write_image(output_image);

    return 0;
}

