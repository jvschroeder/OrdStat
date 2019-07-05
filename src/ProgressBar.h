#include <iostream>
#include <chrono>
#include <ratio>
#include <thread>

class ProgressBar {
public:
    ProgressBar(std::ostream& out, uint64_t maxVal) : maxVal(maxVal)
    {
      start = std::chrono::high_resolution_clock::now();
      last = start;
    }

    void show(uint64_t currentVal, bool forceShow=false) noexcept {
        if(RcppThread::mainThreadID == std::this_thread::get_id() && (forceShow || std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now()-last).count() > 100)) {
          last =  std::chrono::high_resolution_clock::now();
          currentVal = std::min(currentVal,maxVal);
          int progress = (int)((float) currentVal / maxVal * length);
          Rcpp::Rcout << "\r|" << std::string(progress, '=') << std::string(length-progress,' ');
          {
            Rcpp::Rcout << "| " << std::fixed << std::setprecision(1) << 100 * (double)currentVal/maxVal << "% ";
          }
  
          auto elapsed = (std::chrono::high_resolution_clock::now() - start);
          if (currentVal > 0 && currentVal < maxVal) {
              auto remaining = elapsed / ((float)currentVal / maxVal)-elapsed;
              fmt_time(remaining);
              Rcpp::Rcout << " remaining; ";
              
          }
          fmt_time(elapsed);
          Rcpp::Rcout << " elapsed.";
          
          Rcpp::Rcout << std::string(20,' ') << std::flush;
      }
    }
  
private:
  const int length = 40;
  const uint64_t maxVal;
  std::chrono::high_resolution_clock::time_point start,last;
  
  inline void fmt_time(std::chrono::duration<float> dur) {
    Rcpp::Rcout << std::fixed << std::setprecision(2);
    if (dur > std::chrono::hours(1)) {
      Rcpp::Rcout << std::chrono::duration<float, std::ratio<3600>>(dur).count() << "h";
    } else if (dur > std::chrono::minutes(1)) {
      Rcpp::Rcout << std::chrono::duration<float, std::ratio<60>>(dur).count() << "m";
    } else {
      Rcpp::Rcout << std::chrono::duration<float>(dur).count() << "s";
    }
  }
};
