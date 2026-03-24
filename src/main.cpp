//
// Created by  triple-Mu     on 24-01-2023.
// Modified by Q-engineering on 06-03-2024
// Modified by inaciose      on 24-03-2026
//

#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"

using namespace std;
using namespace cv;

#define VIDEO

cv::Size       im_size(640, 640);
const int      num_labels  = 80;
const int      topk        = 100;
const float    score_thres = 0.25f;
const float    iou_thres   = 0.65f;


std::string detections_to_json(const std::vector<Object>& objs, int frame_id)
{
    std::ostringstream oss;
    oss << "{\"frame\":" << frame_id << ",\"objects\":[";

    for (size_t i = 0; i < objs.size(); i++) {
        const auto& o = objs[i];

        float cx = o.rect.x + o.rect.width * 0.5f;
        float cy = o.rect.y + o.rect.height * 0.5f;

        oss << "{"
            << "\"label\":\"" << class_names[o.label] << "\","
            << "\"confidence\":" << o.prob << ","
            << "\"duration\":0,"
            << "\"position\":[" << cx << "," << cy << "],"
            << "\"size\":[" << o.rect.width << "," << o.rect.height << "]"
            << "}";

        if (i < objs.size() - 1) oss << ",";
    }

    oss << "]}";
    return oss.str();
}

int main(int argc, char** argv)
{
    float    f;
    float    FPS[16];
    int      i, Fcnt=0;
    cv::Mat  image;
    std::chrono::steady_clock::time_point Tbegin, Tend;

	// output mode: 0=json, 1=video, 2=json+video
	int outmode = 0;

#ifdef VIDEO

    if (argc < 3) {
        fprintf(stderr,"Usage: ./YoloV8rt model_trt.engine video_id [mode= 0, 1, 2] \n");
        return -1;
    }
    
	if (argc >= 4) {
		std::string arg3 = argv[3];
		if (arg3 == "0" || arg3 == "1" || arg3 == "2")
			outmode = std::stoi(arg3);
	}
	
    int frame_id = 0;
    // TODO change to argunent
    // 2, is 1 on 3, 3 is 1 on 4, 4 is 1 on 5, etc
    int skip = 3;

#else

    if (argc < 3) {
        fprintf(stderr,"Usage: ./YoloV8rt model_trt.engine image_name \n");
        return -1;
    }
    
#endif // VIDEO 
   
    
	cout << "YoloV8r v1.0" << endl;
	
    const string engine_file_path = argv[1];
    const string imagepath = argv[2];

    for(i=0;i<16;i++) FPS[i]=0.0;

    cout << "Set CUDA" << endl;

    cudaSetDevice(0);

    cout << "Loading TensorRT model " << engine_file_path << endl;
    cout << "Wait a second...." << std::flush;
    auto yolov8 = new YOLOv8(engine_file_path);

    cout << "\nLoading the pipe... " << string(10, ' ')<< "" ;
    cout << endl;
    yolov8->MakePipe(true);

#ifdef VIDEO

    cout << "Starting video..." << endl;
    
    VideoCapture cap(imagepath);
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the stream " << imagepath << endl;
        return 0;
    }
    
#endif // VIDEO

    while(1){
		
#ifdef VIDEO

		// json for llama visual context
        static int frame_id = 0;
        
        cap >> image;
        if (image.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }
#else
        image = cv::imread(imagepath);
#endif
        yolov8->CopyFromMat(image, im_size);

        std::vector<Object> objs;

        Tbegin = std::chrono::steady_clock::now();
        yolov8->Infer();
        Tend = std::chrono::steady_clock::now();

        yolov8->PostProcess(objs, score_thres, iou_thres, topk, num_labels);
        yolov8->DrawObjects(image, objs);

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(image, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));


#ifdef VIDEO

		// json for llama visual context
		if(outmode == 0 || outmode == 2) {			
			frame_id++;
			if(frame_id % (skip + 1) == 0) {
				std::string json = detections_to_json(objs, frame_id++);
				std::cout << json << std::endl;
			}
		}

#endif

		if(outmode == 1 || outmode == 2) {
			//imwrite("./out.jpg", image);
			//show output 
			imshow("Jetson Orin Nano- 8 Mb RAM", image);
			char esc = cv::waitKey(1);
			if(esc == 27) break;
		}
    }
    cv::destroyAllWindows();

    delete yolov8;

    return 0;
}
