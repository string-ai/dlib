// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example shows how to use the correlation_tracker from the dlib C++ library.  This
    object lets you track the position of an object as it moves from frame to frame in a
    video sequence.  To use it, you give the correlation_tracker the bounding box of the
    object you want to track in the current video frame.  Then it will identify the
    location of the object in subsequent frames.

    In this particular example, we are going to run on the video sequence that comes with
    dlib, which can be found in the examples/video_frames folder.  This video shows a juice
    box sitting on a table and someone is waving the camera around.  The task is to track the
    position of the juice box as the camera moves around.
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/data_io.h>

using namespace dlib;
using namespace std;
using namespace cv;

void check_user_input();

void update_trackers(cv_image<bgr_pixel> &image, std::vector<correlation_tracker> &vector);

void save_to_dataset(cv_image<bgr_pixel> &image, std::vector<dlib::rectangle> &boxes, int idx);

std::vector<dlib::rectangle> detect_new_people(cv_image<bgr_pixel> &image);

double calc_distance(Point& point_a, Point& point_b) {
    double x_diff = point_a.x - point_b.x;
    double y_diff = point_a.y - point_b.y;
    return sqrt((x_diff * x_diff) + (y_diff * y_diff));
}

bool match_any_tracker(dlib::rectangle &rect, std::vector<correlation_tracker>& trackers,
                       double min_distance = 0.0,
                       double max_distance = 90.0) {
    Point origin(rect.left() + (rect.width()/2), rect.top() + (rect.height()/2) );
    for(auto& tracker : trackers) {
        auto p = tracker.get_position();
        dlib::rectangle p_rec(p.left(), p.top(), p.width(), p.height());

        Point center(p.left() + (p.width()/2), p.top() + (p.height()/2) );

        double distance = calc_distance(origin, center);

        bool match = distance >= min_distance &&
                distance <= max_distance &&
                p_rec.intersect(rect).area() > rect.area() * 0.10;

        if(match){
            return true;
        }
    }
    return false;
}

void update_trackers(cv_image<bgr_pixel> &image, std::vector <correlation_tracker> &trackers, int idx,
        double min_confidence = 0.6) {

    std::vector<drectangle> boxes;

    for(auto& tracker : trackers) {
        double confidence = tracker.update(image);
        if(confidence > min_confidence) {
            boxes.push_back(tracker.get_position());
        }
    }
    if(boxes.size() > 0) {
//        save_to_dataset(image, boxes, idx);
    }
}

string dataset_folder = "/home/rafael/Developer/DataSets/Airgos/automagic";
void save_to_dataset(cv_image<bgr_pixel> &image, std::vector<dlib::rectangle> &boxes, int idx) {


    string image_path = dataset_folder + "/" + "img_" + to_string(idx) + ".jpg";
    save_jpeg(image, image_path);

    string ds_path = dataset_folder + "/" + "ds.xml";
    dlib::image_dataset_file ds_file(ds_path);
    dlib::image_dataset_metadata::dataset meta_data;

    ifstream f(ds_path);
    if(f.good()) {
        load_image_dataset_metadata(meta_data, ds_file.get_filename());
    }
    dlib::image_dataset_metadata::image image_meta(image_path);
    for (auto& drect : boxes) {
        //dlib::rectangle rect = rect_to_rectangle(rect);
        dlib::image_dataset_metadata::box box(drect);
        image_meta.boxes.push_back(box);
    }
    meta_data.images.push_back(image_meta);
    save_image_dataset_metadata(meta_data, ds_file.get_filename());

}

int main(int argc, char** argv) {
    try {
        string command = "exec rm -r " + dataset_folder + "/*";
        system(command.c_str());

        typedef dlib::hashed_feature_image<dlib::hog_image<12, 3, 1, 4, dlib::hog_signed_gradient, dlib::hog_full_interpolation> > feature_extractor_type;
        typedef dlib::scan_image_pyramid<dlib::pyramid_down<3>, feature_extractor_type> hashed_image_scanner_type;
        dlib::object_detector<hashed_image_scanner_type> detector;
        dlib::deserialize("/home/rafael/Developer/string/PeopleCounter/models/hashed_image.dat") >> detector;
        hashed_image_scanner_type scanner;
        hashed_image_scanner_type::feature_vector_type weights = detector.get_w();

        double thresh = 0.5;

        cv::VideoCapture video("/home/rafael/Developer/string/PeopleCounter/tests/videos/test_01_11.mp4");

        cv::Mat frame;
        std::vector<correlation_tracker> trackers;
        int idx = 0;
        while(video.read(frame)) {
            cv_image<bgr_pixel> image = cv_image<bgr_pixel>(frame);
            array2d<unsigned char> gray_image;
            assign_image(gray_image, image);

            idx++;

            //update_trackers(image, trackers, idx);

            std::vector<dlib::rectangle> dets = detect_new_people(image);

            //std::vector<rectangle> dets = detector(gray_image);

//            std::vector<std::pair<double, rectangle> > dets;
//            scanner.load(gray_image);
//            scanner.detect(weights, dets, thresh);

//            double min_area = 150 * 150;

            if(dets.size() > 0) {
                save_to_dataset(image, dets, idx);


//                for(auto& det : dets){
//                    if(det.left() > 0 && det.top() > 0 &&
//                            det.area() >= min_area &&
//                            !match_any_tracker(det, trackers)){
//                        correlation_tracker tracker;
//                        tracker.start_track(image, det);
//                        trackers.push_back(tracker);
//                    }
//                }
            }
        }

    }
    catch (std::exception& e)
    {
        cout << e.what() << endl;
    }
}
/*

 "opts": {
        "history": 500,
        "var_threshold": 60,
        "detect_shadows": true,
        "shadow_value": 0,
        "learning_rate": -1,
        "complexity_reduction_threshold": 0.4,
        "blur_size": 10,
        "threshold_sensitivity": 50,
        "min_height": 80,
        "min_width": 80
      }

 * */

Ptr<BackgroundSubtractorMOG2> _background_subtractor;
std::vector<dlib::rectangle> detect_new_people(cv_image<bgr_pixel> &image) {
    double learning_rate = -1;

    if(! _background_subtractor){
        int history = 500;
        int var_threshold = 60;
        bool detect_shadows = true;
        double complexity_reduction_threshold = 0.4;
        int shadow_value = 0;

        _background_subtractor = createBackgroundSubtractorMOG2(history, var_threshold, detect_shadows);

        _background_subtractor->setComplexityReductionThreshold(complexity_reduction_threshold);
        _background_subtractor->setShadowValue(shadow_value);
    }

    Mat input, output;
    input = toMat(image);
    _background_subtractor->apply(input, output, learning_rate);

    int blur_size = 10;
    int threshold_sensitivity = 50;

    erode(output, output, Mat());
    blur(output, output, Size(blur_size, blur_size));

    threshold(output, output, threshold_sensitivity, 255, THRESH_BINARY);

    int min_height = 80;
    int min_width = 80;

    //these two vectors needed for output of findContours
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    findContours(output,
                 contours,
                 hierarchy,
                 CV_RETR_TREE,
                 CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

    // Approximate contours to polygons + get bounding rects and circles
    std::vector<std::vector<Point> > contours_poly( contours.size() );
    std::vector<dlib::rectangle> detections;
    for( int i = 0; i < contours.size(); i++ ) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true );
        Rect rect = boundingRect(Mat(contours_poly[i]));
        if(rect.width >= min_width && rect.height >= min_height){
            detections.push_back(dlib::rectangle(rect.x, rect.y, rect.width, rect.height));
        }
    }
    return detections;
}