//
// Created by ubuntu on 1/20/23.
//
#include "chrono"
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"
#include "BYTETracker.h"
#include "STrack.h"
using namespace std;

const std::vector<std::string> CLASS_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
	{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
	{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
	{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
	{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
	{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
	{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
	{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
	{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
	{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
	{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
	{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
	{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
	{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
	{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
	{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
	{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
	{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
	{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
	{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
	{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
	{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
	{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
	{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
	{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
	{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
	{ 80, 183, 189 }, { 128, 128, 0 }
};

int main(int argc, char** argv)
{
	// cuda:0
	cudaSetDevice(0);

	const std::string engine_file_path{ argv[1] };
	const std::string path{ argv[2] };

	std::vector<std::string> imagePathList;
	bool isVideo{ false };

	assert(argc == 3);

	auto yolov8 = new YOLOv8(engine_file_path);
	auto tracker = new BYTETracker(15,15);
	yolov8->make_pipe(true);

	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (
			suffix == "jpg" ||
				suffix == "jpeg" ||
				suffix == "png"
			)
		{
			imagePathList.push_back(path);
		}
		else if (
			suffix == "mp4" ||
				suffix == "avi" ||
				suffix == "m4v" ||
				suffix == "mpeg" ||
				suffix == "mov" ||
				suffix == "mkv"
			)
		{
			isVideo = true;
		}
		else
		{
			printf("suffix %s is wrong !!!\n", suffix.c_str());
			std::abort();
		}
	}
//	else if (IsFolder(path))
//	{
//		cv::glob(path + "/*.jpg", imagePathList);
//	}

	cv::Mat res, image;
	cv::Size size = cv::Size{ 640, 640};
	std::vector<Object> objs;	
	int num_frames = 0;
	int total_ms = 0;
	

	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	if (isVideo)
	{
		cv::VideoCapture cap(path);
		if (!cap.isOpened())
		{
			printf("can not open %s\n", path.c_str());
			return -1;
		}
		int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  		int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width,frame_height));

		while (cap.read(image))
		{
			num_frames ++;
			objs.clear();
			yolov8->copy_from_Mat(image, size);
			auto start = std::chrono::system_clock::now();
			yolov8->infer();
			auto end = std::chrono::system_clock::now();

            printf("infer time cost %2.4lf ms\n", chrono::duration_cast<chrono::microseconds>(end - start).count()/ 1000.);


			yolov8->postprocess(objs);
			auto trackerstart = std::chrono::system_clock::now();
			vector<STrack> output_stracks = tracker->update(objs);
			auto trackerend = std::chrono::system_clock::now();

            printf("tracker cost %2.4lf ms\n", chrono::duration_cast<chrono::microseconds>(trackerend - trackerstart).count()/ 1000.);


			total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(trackerend - start).count();

			for (int i = 0; i < output_stracks.size(); i++)
			{
				vector<float> tlwh = output_stracks[i].tlwh;
				bool vertical = tlwh[2] / tlwh[3] > 1.6;
				
				if (tlwh[2] * tlwh[3] > 20 && !vertical)
				{
					Scalar s = tracker->get_color(output_stracks[i].track_id);
					cv::putText(image, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
                	cv::rectangle(image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
					if (output_stracks[i].track_id == 1){
						cout << "is activated " << output_stracks[i].is_activated << endl;
						cout << "Track ID " << output_stracks[i].track_id << endl;
						cout << "State " << output_stracks[i].state << endl;
						cout << "Frame " << output_stracks[i].frame_id << endl;
						cout << "Tracklet Len " << output_stracks[i].tracklet_len << endl;
						cout << "Start Frame " << output_stracks[i].start_frame << endl;
						cout << "Score " << output_stracks[i].score << endl;
					}
				}
				cv::putText(image, cv::format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
			}

            auto allend = std::chrono::system_clock::now();

			auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(allend - start).count() / 1000.;
			printf("all time cost %2.4lf ms\n", tc);

			cv::imshow("result", image);
			video.write(image);
			if (cv::waitKey(10) == 'q')
			{
				break;
			}
		}
			video.release();

	}
	else
	{
		for (auto& path : imagePathList)
		{
			objs.clear();
			image = cv::imread(path);
			yolov8->copy_from_Mat(image, size);
			auto start = std::chrono::system_clock::now();
			yolov8->infer();
			auto end = std::chrono::system_clock::now();
			yolov8->postprocess(objs);
			auto trackerstart = std::chrono::system_clock::now();
			vector<STrack> output_stracks = tracker->update(objs);
			auto trackerend = std::chrono::system_clock::now();
		for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				//int s = output_stracks[i].track_id;
				//cout << " ID " << output_stracks[i].track_id << endl;
				// yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS,output_stracks[i].track_id);
				//cout << " ID is " << output_stracks[i].track_id << endl;		
			}
		}
			// yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS,0);
			//yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
			auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("Detection cost %2.4lf ms\n", tc);
			auto tt = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(trackerend - trackerstart).count() / 1000.;
			printf("Tracker cost %2.4lf ms\n", tt);

			cv::imshow("result", image);
//			cv::imwrite("result.png",image);
//			cv::waitKey(1000/15);
		}
	}
	cv::destroyAllWindows();
	delete yolov8;
	return 0;
}
