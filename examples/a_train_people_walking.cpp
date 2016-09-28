/*
    1- read the files from prid_2011 dataset
    2- train object detector
    3- train shape detector

*/


#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>

#include <cv.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

inline int random_range(int min, int max) {
    static int seed_increment = 1;
    const int seed = (unsigned)++seed_increment * time(NULL);
    std::srand(seed);
    const int rand_value = (unsigned)std::rand();
    const int result = rand_value % max + min;
    if(result < min || result > max) {
        cout << "!!!! random_range out of range :( "
             << " min: " << min
             << " max: " << max
             << " result: " << result << endl;
        return random_range(min, max);
    }
    seed_increment = (unsigned)++seed_increment * result * std::rand();
    return result;
}

inline rectangle random_place_within_boundary(
        rectangle boundary,
        rectangle shape) {
    const int x = random_range(
            boundary.left(),
            boundary.right() - shape.width()
    );
    const int y = random_range(
            boundary.top(),
            boundary.bottom() - shape.height()
    );
    return rectangle(x, y, x + shape.width(), y + shape.height());
}

inline bool has_overlaps(rectangle place, std::vector<rectangle> places) {
    for (int i = 0; i < places.size() ; i++) {
        rectangle existing_place = places[i];
        if(existing_place.contains(place.tl_corner()) ||
                existing_place.contains(place.bl_corner()) ||
                existing_place.contains(place.tr_corner()) ||
                existing_place.contains(place.br_corner())){
            return true;
        }
    }
    return false;
}

std::vector<rectangle> random_places_within_boundary(
        int num_places,
        rectangle boundary,
        rectangle shape){
    const int max_tries = 100;
    std::vector<rectangle> places;
    for (int i = 0; i < num_places ; i++) {
        int tries = 0;
        while(++tries < max_tries) {
            rectangle new_place = random_place_within_boundary(boundary, shape);
            if( ! has_overlaps(new_place, places)){
                places.push_back(new_place);
                break;
            }
        }
    }
    return places;
}

template <
        typename array_type
>
void add_to_background_scene (
        const std::string& bg_imgs_folder,
        array_type& images,
        rectangle boundary,
        std::vector<std::vector<rectangle> >& object_locations,
        std::vector<std::vector<rectangle> >& ignore_object_locations
) {
    // make sure requires clause is not broken
    DLIB_ASSERT( images.size() == object_locations.size() &&
                 images.size() == ignore_object_locations.size(),
                 "\t void add_to_background_scene()"
                         << "\n\t Invalid inputs were given to this function."
                         << "\n\t images.size():   " << images.size()
                         << "\n\t object_locations.size():  " << object_locations.size()
                         << "\n\t ignore_object_locations.size(): " << ignore_object_locations.size()
    );


    typedef typename array_type::value_type image_type;

    //TODO: load random bg images
    image_type bg_image;
    load_image(bg_image, bg_imgs_folder);

    for (unsigned long img_idx = 0; img_idx < images.size(); ++img_idx) {

        std::vector<rectangle> rects, ignore_rects;

        rects = object_locations[img_idx];
        std::vector<rectangle> adjusted_rects;

        ignore_rects = ignore_object_locations[img_idx];
        std::vector<rectangle> adjusted_ignore_rects;

        image_type final_img;
        assign_image(final_img, bg_image);

        std::vector<rectangle> positions = random_places_within_boundary(2, boundary, get_rect(images[img_idx]));

        for (int i = 0; i < positions.size(); ++i) {
            rectangle position = positions[i];

            merge_with_background(
                    images[img_idx],
                    final_img,
                    position,
                    rects,
                    adjusted_rects,
                    ignore_rects,
                    adjusted_ignore_rects);
        }



        //images[img_idx] = final_img;
        swap(final_img, images[img_idx]);

        object_locations[img_idx] = adjusted_rects;
        ignore_object_locations[img_idx] = adjusted_ignore_rects;
    }

}

// ----------------------------------------------------------------------------------------

void pick_best_window_size (
    const std::vector<std::vector<rectangle> >& boxes,
    unsigned long& width,
    unsigned long& height,
    const unsigned long target_size
)
/*!
    ensures
        - Finds the average aspect ratio of the elements of boxes and outputs a width
          and height such that the aspect ratio is equal to the average and also the
          area is equal to target_size.  That is, the following will be approximately true:
            - #width*#height == target_size
            - #width/#height == the average aspect ratio of the elements of boxes.
!*/
{
    // find the average width and height
    running_stats<double> avg_width, avg_height;
    for (unsigned long i = 0; i < boxes.size(); ++i)
    {
        for (unsigned long j = 0; j < boxes[i].size(); ++j)
        {
            avg_width.add(boxes[i][j].width());
            avg_height.add(boxes[i][j].height());
        }
    }

    // now adjust the box size so that it is about target_pixels pixels in size
    double size = avg_width.mean()*avg_height.mean();
    double scale = std::sqrt(target_size/size);

    width = (unsigned long)(avg_width.mean()*scale+0.5);
    height = (unsigned long)(avg_height.mean()*scale+0.5);
    // make sure the width and height never round to zero.
    if (width == 0)
        width = 1;
    if (height == 0)
        height = 1;
}

// ----------------------------------------------------------------------------------------

bool contains_any_boxes (
    const std::vector<std::vector<rectangle> >& boxes
)
{
    for (unsigned long i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i].size() != 0)
            return true;
    }
    return false;
}

// ----------------------------------------------------------------------------------------

void throw_invalid_box_error_message (
    const std::string& dataset_filename,
    const std::vector<std::vector<rectangle> >& removed,
    const unsigned long target_size
)
{
    image_dataset_metadata::dataset data;
    load_image_dataset_metadata(data, dataset_filename);

    std::ostringstream sout;
    sout << "Error!  An impossible set of object boxes was given for training. ";
    sout << "All the boxes need to have a similar aspect ratio and also not be ";
    sout << "smaller than about " << target_size << " pixels in area. ";
    sout << "The following images contain invalid boxes:\n";
    std::ostringstream sout2;
    for (unsigned long i = 0; i < removed.size(); ++i)
    {
        if (removed[i].size() != 0)
        {
            const std::string imgname = data.images[i].filename;
            sout2 << "  " << imgname << "\n";
        }
    }
    throw error("\n"+wrap_string(sout.str()) + "\n" + sout2.str());
}

// ----------------------------------------------------------------------------------------

template<class T>
T base_name(T const & path, T const & delims = "/\\")
{
    return path.substr(path.find_last_of(delims) + 1);
}

template<class T>
T folder_name(T const & path, T const & delims = "/\\")
{
    return path.substr(0, path.find_last_of(delims));
}

template<class T>
T remove_extension(T const & filename)
{
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}

int main(int argc, char** argv)
{  
    try
    {
        command_line_parser parser;
        parser.add_option("h", "Display this help message.");
        parser.add_option("t", "Train an object detector and save the detector to disk.");
        parser.add_option("cross-validate",
                          "Perform cross-validation on an image dataset and print the results.");
        parser.add_option("test", "Test a trained detector on an image dataset and print the results.");
        parser.add_option("u", "Upsample each input image <arg> times. Each upsampling quadruples the number of pixels in the image (default: 0).", 1);
        parser.add_option("model", "Model file name.", 1);

        parser.add_option("bg_img", "Background image used to place training images.", 1);



        parser.set_group_name("training/cross-validation sub-options");
        parser.add_option("v","Be verbose.");
        parser.add_option("folds","When doing cross-validation, do <arg> folds (default: 3).",1);
        parser.add_option("c","Set the SVM C parameter to <arg> (default: 1.0).",1);
        parser.add_option("threads", "Use <arg> threads for training (default: 4).",1);
        parser.add_option("eps", "Set training epsilon to <arg> (default: 0.01).", 1);
        parser.add_option("target-size", "Set size of the sliding window to about <arg> pixels in area (default: 80*80).", 1);
        parser.add_option("flip", "Add left/right flipped copies of the images into the training dataset.  Useful when the objects "
            "you want to detect are left/right symmetric.");

        parser.parse(argc, argv);

        // Now we do a little command line validation.  Each of the following functions
        // checks something and throws an exception if the test fails.
        const char* one_time_opts[] = {"h", "v", "t", "cross-validate", "c", "threads", "target-size",
                                        "folds", "test", "eps", "u", "flip", "model", "bg_img"};
        parser.check_one_time_options(one_time_opts); // Can't give an option more than once
        // Make sure the arguments to these options are within valid ranges if they are supplied by the user.
        parser.check_option_arg_range("c", 1e-12, 1e12);
        parser.check_option_arg_range("eps", 1e-5, 1e4);
        parser.check_option_arg_range("threads", 1, 1000);
        parser.check_option_arg_range("folds", 2, 100);
        parser.check_option_arg_range("u", 0, 8);
        parser.check_option_arg_range("target-size", 4*4, 10000*10000);
        const char* incompatible[] = {"t", "cross-validate", "test"};
        parser.check_incompatible_options(incompatible);
        // You are only allowed to give these training_sub_ops if you also give either -t or --cross-validate.
        const char* training_ops[] = {"t", "cross-validate"};
        const char* training_sub_ops[] = {"v", "c", "threads", "target-size", "eps", "flip"};
        parser.check_sub_options(training_ops, training_sub_ops); 
        parser.check_sub_option("cross-validate", "folds"); 


        if (parser.option("h"))
        {
            cout << "Usage: train_object_detector [options] <image dataset file|image file>\n";
            parser.print_options(); 
                                       
            return EXIT_SUCCESS;
        }


        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        // Get the upsample option from the user but use 0 if it wasn't given.
        const unsigned long upsample_amount = get_option(parser, "u", 0);

        string xml_full_path = parser[0];
        string xml_file_name = base_name(xml_full_path);
        string xml_file_folder = folder_name(xml_full_path);
        string detector_file_name = remove_extension(xml_file_name) + ".svm";

        if (parser.option("model")) {
            detector_file_name = get_option(parser, "model", detector_file_name);
        }

        string detector_full_path = xml_file_folder + "/" + detector_file_name;

        if (parser.option("t") || parser.option("cross-validate"))
        {
            if (parser.number_of_arguments() != 1)
            {
                cout << "You must give an image dataset metadata XML file produced by the imglab tool." << endl;
                cout << "\nTry the -h option for more information." << endl;
                return EXIT_FAILURE;
            }

            dlib::array<array2d<unsigned char> > images;
            std::vector<std::vector<rectangle> > object_locations, ignore;

            cout << "Loading image dataset from metadata file " << parser[0] << endl;

            ignore = load_image_dataset(images, object_locations, parser[0]);

            cout << "Number of images loaded: " << images.size() << endl;

            // Get the options from the user, but use default values if they are not
            // supplied.
            const int threads = get_option(parser, "threads", 4);
            const double C   = get_option(parser, "c", 1.0);
            const double eps = get_option(parser, "eps", 0.01);
            unsigned int num_folds = get_option(parser, "folds", 3);
            const unsigned long target_size = get_option(parser, "target-size", 80*80);
            // You can't do more folds than there are images.  
            if (num_folds > images.size())
                num_folds = images.size();

            // Upsample images if the user asked us to do that.
            for (unsigned long i = 0; i < upsample_amount; ++i)
                upsample_image_dataset<pyramid_down<2> >(images, object_locations, ignore);

            if (parser.option("bg_img")) {

                image_window win1;
                image_window win2;
                image_window win3;

                const string bg_image = get_option(parser, "bg_img", "");

                const rectangle boundary(125, 120, 825, 800);

                add_to_background_scene(bg_image, images, boundary, object_locations, ignore);

                //ignore = load_image_dataset_with_background(images, object_locations, parser[0], bg_image);

                win1.set_image (images[0]);
                win2.set_image (images[1]);
                win3.set_image (images[2]);

                win1.wait_until_closed();
                win2.wait_until_closed();
                win3.wait_until_closed();
            }
            else{

            }

            image_scanner_type scanner;
            unsigned long width, height;
            pick_best_window_size(object_locations, width, height, target_size);
            scanner.set_detection_window_size(width, height); 

            structural_object_detection_trainer<image_scanner_type> trainer(scanner);
            trainer.set_num_threads(threads);
            if (parser.option("v"))
                trainer.be_verbose();
            trainer.set_c(C);
            trainer.set_epsilon(eps);

            // Now make sure all the boxes are obtainable by the scanner.  
            std::vector<std::vector<rectangle> > removed;
            removed = remove_unobtainable_rectangles(trainer, images, object_locations);
            // if we weren't able to get all the boxes to match then throw an error 
            if (contains_any_boxes(removed))
            {
                unsigned long scale = upsample_amount+1;
                scale = scale*scale;
                throw_invalid_box_error_message(parser[0], removed, target_size/scale);
            }

            if (parser.option("flip"))
                add_image_left_right_flips(images, object_locations, ignore);

            if (parser.option("t"))
            {
                // Do the actual training and save the results into the detector object.  
                object_detector<image_scanner_type> detector = trainer.train(images, object_locations, ignore);

                cout << "Saving trained detector to: " << detector_full_path << endl;
                serialize(detector_full_path) << detector;


                cout << "Testing detector on training data..." << endl;
                cout << "Test detector (precision,recall,AP): " << test_object_detection_function(detector, images, object_locations, ignore) << endl;
            }
            else
            {
                // shuffle the order of the training images
                randomize_samples(images, object_locations);

                cout << num_folds << "-fold cross validation (precision,recall,AP): "
                     << cross_validate_object_detection_trainer(trainer, images, object_locations, ignore, num_folds) << endl;
            }

            cout << "Parameters used: " << endl;
            cout << "  threads:                 "<< threads << endl;
            cout << "  C:                       "<< C << endl;
            cout << "  eps:                     "<< eps << endl;
            cout << "  target-size:             "<< target_size << endl;
            cout << "  detection window width:  "<< width << endl;
            cout << "  detection window height: "<< height << endl;
            cout << "  upsample this many times : "<< upsample_amount << endl;
            if (parser.option("flip"))
                cout << "  trained using left/right flips." << endl;
            if (parser.option("cross-validate"))
                cout << "  num_folds: "<< num_folds << endl;
            cout << endl;

            return EXIT_SUCCESS;
        }

        // The rest of the code is devoted to testing an already trained object detector.

        if (parser.number_of_arguments() == 0)
        {
            cout << "You must give an image or an image dataset metadata XML file produced by the imglab tool." << endl;
            cout << "\nTry the -h option for more information." << endl;
            return EXIT_FAILURE;
        }

        cout << "Loading trained detector from: " << detector_full_path << endl;

        // load a previously trained object detector and try it out on some data
        ifstream fin(detector_full_path, ios::binary);
        if (!fin)
        {
            cout << "Can't find a trained object detector file object_detector.svm. " << endl;
            cout << "You need to train one using the -t option." << endl;
            cout << "\nTry the -h option for more information." << endl;
            return EXIT_FAILURE;

        }
        object_detector<image_scanner_type> detector;
        deserialize(detector, fin);

        dlib::array<array2d<unsigned char> > images;
        // Check if the command line argument is an XML file
        if (tolower(right_substr(parser[0],".")) == "xml")
        {
            std::vector<std::vector<rectangle> > object_locations, ignore;
            cout << "Loading image dataset from metadata file " << parser[0] << endl;
            ignore = load_image_dataset(images, object_locations, parser[0]);
            cout << "Number of images loaded: " << images.size() << endl;

            // Upsample images if the user asked us to do that.
            for (unsigned long i = 0; i < upsample_amount; ++i)
                upsample_image_dataset<pyramid_down<2> >(images, object_locations, ignore);

            if (parser.option("test"))
            {
                cout << "Testing detector on data..." << endl;
                cout << "Results (precision,recall,AP): " << test_object_detection_function(detector, images, object_locations, ignore) << endl;
                return EXIT_SUCCESS;
            }
        }
        else
        {
            // In this case, the user should have given some image files.  So just
            // load them.
            images.resize(parser.number_of_arguments());
            for (unsigned long i = 0; i < images.size(); ++i)
                load_image(images[i], parser[i]);

            // Upsample images if the user asked us to do that.
            for (unsigned long i = 0; i < upsample_amount; ++i)
            {
                for (unsigned long j = 0; j < images.size(); ++j)
                    pyramid_up(images[j]);
            }
        }


        // Test the detector on the images we loaded and display the results
        // in a window.
        image_window win;
        for (unsigned long i = 0; i < images.size(); ++i)
        {
            // Run the detector on images[i] 
            const std::vector<rectangle> rects = detector(images[i]);
            cout << "Number of detections: "<< rects.size() << endl;

            // Put the image and detections into the window.
            win.clear_overlay();
            win.set_image(images[i]);
            win.add_overlay(rects, rgb_pixel(255,0,0));

            cout << "Hit enter to see the next image.";
            cin.get();
        }


    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        cout << "\nTry the -h option for more information." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

