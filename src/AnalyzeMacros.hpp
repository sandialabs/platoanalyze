/*
 * AnalyzeMacros.hpp
 *
 *  Created on: June 30, 2019
 */

#pragma once

#include <string>

namespace Plato
{

#define REPORT(msg) \
        std::cout << std::string("\nANALYZE REPORT: ") + msg + "\n";

#define WARNING(msg) \
        std::cout << std::string("\n\nFILE: ") + __FILE__ \
        + std::string("\nFUNCTION: ") + __PRETTY_FUNCTION__ \
        + std::string("\nLINE:") + std::to_string(__LINE__) \
        + std::string("\nMESSAGE: ") + msg + "\n\n";

#define ANALYZE_PRINTERR(msg) \
        std::cout << std::string("\n\nFILE: ") + __FILE__ \
        + std::string("\nFUNCTION: ") + __PRETTY_FUNCTION__ \
        + std::string("\nLINE:") + std::to_string(__LINE__) \
        + std::string("\nMESSAGE: ") + msg + "\n\n";

#define ANALYZE_THROWERR(msg) \
        throw std::runtime_error(std::string("\n\nFILE: ") + __FILE__ \
        + std::string("\nFUNCTION: ") + __PRETTY_FUNCTION__ \
        + std::string("\nLINE:") + std::to_string(__LINE__) \
        + std::string("\nMESSAGE: ") + msg + "\n\n");

#define GPU_WARNING(msg) \
  printf("\n\nFILE: %s \nFUNCTION: %s \nLINE: %d \nMESSAGE: %s",  \
         __FILE__, __PRETTY_FUNCTION__, __LINE__, msg );
}
//namespace Plato
