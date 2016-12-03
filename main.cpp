// Copyright (c) Tamas Csala
// based on http://jponttuset.cat/solving-sudokus-like-a-pro-1/

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <vector>
#include <unordered_set>
#include <string>
#include <map>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>

/* Tesseract OCR includes */
#include <tesseract/baseapi.h>

#include "sudoku.hpp"

enum FunctionMode {
  kDisplayUnchangedVideo = 0,
  kCannyEdgeDetect = 1,
  kHoughTransform = 2,
  kShowCornerPoints = 3,
  kNumberCells = 4,
  kRecognizedNumbers = 5,
  kShowSolution = 6,
  kLast = kShowSolution
} mode = kShowSolution;

constexpr double kEpsilon = 1e-8;

cv::Mat Canny(const cv::Mat& frame_gray) {
  constexpr int kRatio = 3;
  constexpr int kKernelSize = 3;
  constexpr int kLowThreshold = 30;

  cv::Mat blurred_gray, edges;
  cv::blur(frame_gray, blurred_gray, cv::Size(3,3));
  cv::Canny(blurred_gray, edges, kLowThreshold, kLowThreshold*kRatio, kKernelSize);
  return edges;
}

bool input_freeze = false;

struct Line {
  // start and end points
  cv::Point e1;
  cv::Point e2;

  size_t id; // index in the container returned by cv::HoughLines
  size_t hv_id; // index in the horizontal or vertical container

  // Intersections with orthogonal lines
  std::multimap<double,size_t> intersections;
};

/* Finds the intersection of two lines, or returns false.
 *  - The lines are defined by (o1, p1) and (o2, p2).
 *  - The intersection point is returned in 'inters'
 */
bool Intersection(cv::Point2f o1, cv::Point2f p1,
                  cv::Point2f o2, cv::Point2f p2, cv::Point2f &inters) {
  cv::Point2f x = o2 - o1;
  cv::Point2f d1 = p1 - o1;
  cv::Point2f d2 = p2 - o2;

  float cross = d1.x*d2.y - d1.y*d2.x;
  if (std::abs(cross) < kEpsilon) {
    return false;
  }

  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  inters = o1 + d1 * t1;
  return true;
}

/* Function that takes two sets of lines, and looks for a pattern of ten evenly-distributed lines at the first set,
 * with respect to the intersections with the other set of lines.
 * - lines1 and lines2 are input vectors containing pairs with the 'Line' struct defining the lines, and their distance to the origin
 * - sel_lines is the output vector containing the ids of the ten 'recognized' sets of lines
 *   (e.g., all ids in sel_lines[0] are part of the left-most or upper line of the grid)
 * - The function returns true only if it finds an acceptable pattern of ten lines
 */
bool ClassifyLines(const std::vector<std::pair<double, Line>>& lines1,
                   const std::vector<std::pair<double, Line>>& lines2,
                   std::vector<std::set<size_t>>& sel_lines)
{
  // Struct that helps us store and sort pairs of lines
  struct PairStruct {
    PairStruct(size_t id1 = 0, size_t id2 = 0, double inters = 0)
      : id1(id1), id2(id2), inters(inters) { }
    size_t id1;
    size_t id2;
    double inters;
  };

  // At least 20 pixels apart between lines of different sets (coming from different *true* lines)
  double dist_th = 20;

  if (lines1.empty()) {
    return false;
  } else {
    // We store the distance between consecutive lines, to look for nine similar 'distances'
    std::vector<std::pair<double, PairStruct>> int_diffs;

    // Get the line in the middle of the detected lines
    size_t horiz_id = round(lines1.size() / 2);
    auto line_it = lines1.begin() + horiz_id;
    const Line& middle_line = line_it->second;

    // The line in the middle intersects with less than 10 lines, no Sudoku grid in the image
    if (middle_line.intersections.size() <= 9) {
      return false;
    } else {
      // Scan all pairs of consecutive intersections with the middle line and store the 'pair'
      auto prev_inter = middle_line.intersections.begin();
      auto inter = middle_line.intersections.begin();
      ++inter;
      for (; inter != middle_line.intersections.end(); ++inter, ++prev_inter) {
        int_diffs.push_back(std::make_pair(inter->first - prev_inter->first,
            PairStruct{prev_inter->second, inter->second, inter->first}));
      }

      // Sort the pairs of consecutive intersections with respect to their distance
      auto compare = [](const std::pair<double, PairStruct> &left,
                        const std::pair<double, PairStruct> &right) {
        return left.first < right.first;
      };
      std::sort(int_diffs.begin(), int_diffs.end(), compare);

      // Look for the round of 9 most similar differences
      auto it1 = int_diffs.begin();
      auto it2 = int_diffs.begin() + 8;
      double min_diff = 1000000;
      int min_ind = -1;
      size_t curr_ind = 0;
      for (; it2 < int_diffs.end(); ++it1, ++it2, ++curr_ind) {
        if (it1->first > dist_th) {
          if (it2->first - it1->first < min_diff) {
            min_diff = it2->first - it1->first;
            min_ind  = (int)curr_ind;
          }
        }
      }

      // Have we found a 'round'?
      if (min_ind < 0) {
        return false;
      } else if (std::max(int_diffs[min_ind].first, int_diffs[min_ind+8].first) /
                 std::min(int_diffs[min_ind].first, int_diffs[min_ind+8].first) > 1.3) {
        return false;
      } else {
        // Put them together to sort them
        std::vector<PairStruct> sel_pairs(9);
        for (size_t i = 0; i < 9; ++i) {
          sel_pairs[i] = int_diffs[min_ind+i].second;
        }
        auto compare = [](const PairStruct &left, const PairStruct &right) {return left.inters < right.inters;};
        std::sort(sel_pairs.begin(), sel_pairs.end(), compare);

        // Start the sets of similar lines
        sel_lines.resize(10);
        for (size_t i = 0; i < 9; ++i) {
          sel_lines[i].insert(sel_pairs[i].id1);
          sel_lines[i+1].insert(sel_pairs[i].id2);
        }
      }
    }
  }

  return true;
}

cv::Point2f MeanIntersection(const std::set<size_t>& h_set,
                             const std::set<size_t>& v_set,
                             const std::vector<std::pair<double, Line>>& horiz,
                             const std::vector<std::pair<double, Line>>& verti)
{
    // Get all interesections
    std::vector<cv::Point2f> all_int;
    for (auto h_it : h_set) {
      for(auto v_it : v_set) {
        cv::Point2f inters;
        if (Intersection(horiz.at(h_it).second.e1, horiz.at(h_it).second.e2,
                         verti.at(v_it).second.e1, verti.at(v_it).second.e2, inters)) {
          all_int.push_back(inters);
        }
      }
    }

    // Get the mean
    cv::Point2f mean = all_int[0];
    for (size_t i = 1; i < all_int.size(); ++i) {
      mean = mean + all_int[i];
    }
    mean.x = mean.x / (float)all_int.size();
    mean.y = mean.y / (float)all_int.size();
    return mean;
}


void DrawPoint(cv::Mat& img, cv::Point center) {
  int thickness = -1;
  int lineType = 8;
  cv::circle(img, center, 4, cv::Scalar(0, 255, 255), thickness, lineType);
}

unsigned int RecognizeDigit(cv::Mat& im, tesseract::TessBaseAPI& tess)
{
    tess.SetImage((uchar*)im.data, im.size().width, im.size().height, im.channels(), (int)im.step1());
    tess.Recognize(0);
    const char* out = tess.GetUTF8Text();
    if (out)
        if(out[0]=='1' or out[0]=='I' or out[0]=='i' or out[0]=='/' or out[0]=='|' or out[0]=='l' or out[0]=='t')
            return 1;
        else if(out[0]=='2')
            return 2;
        else if(out[0]=='3')
            return 3;
        else if(out[0]=='4')
            return 4;
        else if(out[0]=='5' or out[0]=='S' or out[0]=='s')
            return 5;
        else if(out[0]=='6')
            return 6;
        else if(out[0]=='7')
            return 7;
        else if(out[0]=='8')
            return 8;
        else if(out[0]=='9')
            return 9;
        else
            return 0;
    else
        return 0;
}


int main() {
  cv::VideoCapture capture("sudoku.mp4");
  std::string window_name = "AR Sudoku Solver";
  cv::namedWindow(window_name, CV_WINDOW_KEEPRATIO);

  tesseract::TessBaseAPI tess;
  if (tess.Init(nullptr, "eng")) {
    std::cerr << "Could not initialize tesseract." << std::endl;
    return 1;
  }

  cv::Mat frame, last_frame;
  while (true) {
    if (input_freeze) {
      frame = last_frame.clone();
    } else {
      capture >> frame;
      last_frame = frame.clone();
    }

    if (frame.empty()) {
      std::cout << "No camera input... exiting now." << std::endl;
      return 0;
    }

    if (mode == kDisplayUnchangedVideo) {
      cv::imshow(window_name, frame);
    } else {
      cv::Mat frame_gray;
      cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
      cv::Mat edges = Canny(frame_gray);
      if (mode == kCannyEdgeDetect) {
        imshow(window_name, edges);
      } else {
        std::vector<cv::Vec2f> detected_lines;
        cv::HoughLines(edges, detected_lines, 2, CV_PI/180, 300, 0, 0);

        // Detected lines -> drawable segments
        std::vector<Line> lines(detected_lines.size());
        for (size_t i = 0; i < detected_lines.size(); ++i) {
          float rho = detected_lines[i][0], theta = detected_lines[i][1];
          double a = std::cos(theta), b = std::sin(theta);
          double x0 = a*rho, y0 = b*rho;
          // make the end points stay outside the picture
          lines[i].e1.x = cvRound(x0 + 2000*(-b));
          lines[i].e1.y = cvRound(y0 + 2000*(a));
          lines[i].e2.x = cvRound(x0 - 2000*(-b));
          lines[i].e2.y = cvRound(y0 - 2000*(a));
          lines[i].id = i;
        }

        // Separate them into horizontal and vertical by setting a threshold on the slope
        std::vector<std::pair<double, Line>> horizontal_lines;
        std::vector<std::pair<double, Line>> vertical_lines;
        for (size_t i = 0; i < lines.size(); ++i) {
          if (detected_lines[i][1] < CV_PI/20 || detected_lines[i][1] > CV_PI - CV_PI/20) {
            // Vertical if close to 180 deg or to 0 deg
            vertical_lines.push_back(std::make_pair(detected_lines[i][0], lines[i]));
          } else if (std::abs(detected_lines[i][1] - CV_PI/2) < CV_PI/20) {
            // Horizontal if close to 90 deg
            horizontal_lines.push_back(std::make_pair(detected_lines[i][0], lines[i]));
          }
        }

        // Sort them by distance from origin (rho)
        auto compare = [](const std::pair<double, Line> &left, const std::pair<double, Line> &right) {return left.first < right.first;};
        std::sort(vertical_lines.begin(), vertical_lines.end(), compare);
        std::sort(horizontal_lines.begin(), horizontal_lines.end(), compare);

        for (size_t i = 0; i < vertical_lines.size(); ++i) {
          vertical_lines[i].second.hv_id = i;
        }
        for (size_t i = 0; i < horizontal_lines.size(); ++i) {
          horizontal_lines[i].second.hv_id = i;
        }

        if (mode == kHoughTransform) {
          for (auto line : horizontal_lines) {
            cv::line(frame, line.second.e1, line.second.e2, cv::Scalar(0, 0, 255), 3, CV_AA);
          }
          for (auto line : vertical_lines) {
            cv::line(frame, line.second.e1, line.second.e2, cv::Scalar(0, 255, 0), 3, CV_AA);
          }
        } else {
          size_t sx = frame.cols;
          size_t sy = frame.rows;

          // Compute pairwise intersections between vertical and horizontal lines
          for (auto& vert_it : vertical_lines) {
            for (auto& hori_it : horizontal_lines) {
              auto& vertical_line = vert_it.second;
              auto& horizontal_line = hori_it.second;
              cv::Point2f inters;
              if (Intersection(vertical_line.e1, vertical_line.e2,
                               horizontal_line.e1, horizontal_line.e2, inters)) {
                if (0 <= inters.x && inters.x < sx && 0 <= inters.y && inters.y < sy) {
                    vertical_line.intersections.insert(std::make_pair(inters.y, horizontal_line.hv_id));
                    horizontal_line.intersections.insert(std::make_pair(inters.x, vertical_line.hv_id));
                }
              }
            }
          }

          // Scan one line in the center (less likely to be erroneous) and classify the orthogonal lines
          std::vector<std::set<size_t>> sel_v;
          bool good1 = ClassifyLines(horizontal_lines, vertical_lines, sel_v);

          std::vector<std::set<size_t>> sel_h;
          bool good2 = ClassifyLines(vertical_lines, horizontal_lines, sel_h);

          if (good1 && good2) {
            std::vector<std::vector<cv::Point2f>> corners(10, std::vector<cv::Point2f>(10));
            for (size_t i = 0; i < 10; ++i) {
              for (size_t j = 0; j < 10; ++j) {
                corners[i][j] = MeanIntersection(sel_h[i], sel_v[j], horizontal_lines, vertical_lines);
              }
            }

            if (mode == kShowCornerPoints) {
              for (size_t i = 0; i < 10; ++i) {
                for (size_t j = 0; j < 10; ++j){
                  DrawPoint(frame, corners[i][j]);
                }
              }
            } else {
              // Create the boxes of the cells
              float reduce_percent = 0.6;
              std::vector<std::vector<std::pair<cv::Point2f, cv::Point2f>>> boxes(9, std::vector<std::pair<cv::Point2f, cv::Point2f>>(9));
              for (size_t i = 0; i < 9; ++i) {
                for (size_t j = 0; j < 9; ++j) {
                  cv::Point2f ul = corners[i][j];
                  cv::Point2f dr = corners[i+1][j+1];

                  /* We reduce the size a certain percentage to avoid borders */
                  float w = (dr.x - ul.x) * reduce_percent;
                  float h = (dr.y - ul.y) * reduce_percent;
                  float c_x = (dr.x + ul.x)/2;
                  float c_y = (dr.y + ul.y)/2;
                  ul.x = c_x - w/2;
                  ul.y = c_y - h/2;
                  dr.x = c_x + w/2;
                  dr.y = c_y + h/2;

                  boxes[i][j].first = ul;
                  boxes[i][j].second = dr;
                }
              }

              if (mode == kNumberCells) {
                for(size_t i = 0; i < 9; ++i) {
                  for(size_t j = 0; j < 9; ++j) {
                    cv::rectangle(frame, boxes[i][j].first, boxes[i][j].second, cv::Scalar(255, 255, 0));
                  }
                }
              } else {
                // Get the image of the Sudoku full box by getting the first and last grids
                //  - ulx: Up Left X
                //  - uly: Up Left Y
                //  - drx: Down Right X
                //  - dry: Down Right Y
                //
                unsigned int ulx = round(std::min(corners[0][0].x,corners[9][0].x));
                unsigned int uly = round(std::min(corners[0][0].y,corners[0][9].y));

                unsigned int drx = round(std::max(corners[0][9].x,corners[9][9].x));
                unsigned int dry = round(std::max(corners[9][0].y,corners[9][9].y));

                // This is to be robust against some degenerate cases
                if (sx < ulx || sy < uly || sx < drx || sy < dry)
                  continue;

                // Crop the image
                cv::Mat sudoku_box(frame_gray, cv::Rect(ulx, uly, drx-ulx, dry-uly));

                // Apply local thresholding
                cv::Mat sudoku_th = sudoku_box.clone();
                cv::adaptiveThreshold(sudoku_box, sudoku_th, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);

                // Process all boxes and classify whether they are empty (we'll say 0) or there is a number 1-9
                std::vector<std::vector<unsigned int>> rec_digits(9, std::vector<unsigned int>(9));
                for (size_t i = 0; i < 9; ++i) {
                  for (size_t j = 0; j < 9; ++j) {
                    // Get the square as an image
                    cv::Mat digit_box(sudoku_th, cv::Rect(
                      round(boxes[i][j].first.x)-ulx, round(boxes[i][j].first.y)-uly,
                      round(boxes[i][j].second.x-boxes[i][j].first.x),
                      round(boxes[i][j].second.y-boxes[i][j].first.y)));

                    // Recognize the digit using the OCR
                    rec_digits[i][j] = RecognizeDigit(digit_box, tess);
                  }
                }

                if (mode == kRecognizedNumbers) {
                  for (size_t i = 0; i < 9; ++i) {
                    for (size_t j = 0; j<  9; ++j) {
                      if (rec_digits[i][j] != 0) {
                        cv::Point text_pos(boxes[i][j].first.x+(boxes[i][j].second.x-boxes[i][j].first.x)/5,
                                           boxes[i][j].second.y-(boxes[i][j].second.y-boxes[i][j].first.y)/5);
                        std::stringstream ss;
                        ss << (int)rec_digits[i][j];
                        putText(frame, ss.str(), text_pos, CV_FONT_HERSHEY_DUPLEX, /*Size*/1,
                                cv::Scalar(0, 255, 0), /*Thickness*/ 1, 8);
                      }
                    }
                  }
                } else {
                  constexpr int N = 3;
                  Sudoku<N> sudoku;

                  for (size_t i = 0; i < N*N; ++i) {
                    for (size_t j = 0; j < N*N; ++j) {
                      sudoku.set_value(i, j, rec_digits[i][j]);
                    }
                  }

                  if (sudoku.solve()) {
                    for (size_t i = 0; i < N*N; ++i) {
                      for (size_t j = 0; j < N*N; ++j) {
                        if (rec_digits[i][j] == 0) {
                            cv::Point text_pos(boxes[i][j].first.x +(boxes[i][j].second.x-boxes[i][j].first.x)/5,
                                               boxes[i][j].second.y-(boxes[i][j].second.y-boxes[i][j].first.y)/5);
                            std::stringstream ss;
                            ss << (int)sudoku.get_value(i, j);
                            cv::putText(frame, ss.str(), text_pos, CV_FONT_HERSHEY_DUPLEX, /*Size*/1,
                                    cv::Scalar(255, 0, 0), /*Thickness*/ 1, 8);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        imshow(window_name, frame);
      }
    }

    char key = (char)cv::waitKey(5);
    if ('0' <= key && key <= '0' + kLast) {
      mode = static_cast<FunctionMode>(key - '0');
    } else if (key == 27) { // Escape
      return 0;
    } else if (key == ' ') {
      input_freeze = true;
    } else if (key == '\n') {
      input_freeze = false;
    }
  }
}
