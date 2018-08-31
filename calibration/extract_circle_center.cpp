#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;

vector<vector<Point2f>> extract_circle_center(vector<Mat> images);//函数声明,实体在后边

int main() {
	string prefix;
	string type;
		vector<String> filenames;
	vector<Mat> images;

	cout << "specify the image prefix" << endl;
	cin >> prefix;
	cout << "specify the image type" << endl;
	cin >> type;
	string path = "./image/" + prefix + "*." + type;
	glob(path, filenames);//将指定路径下的文件名存入filenames向量

	for (String filename:filenames) {  //依次读入指定图片，并存入images向量中
		cout << "reading " << filename << endl;
		Mat img = imread(filename,0);
		if (img.empty()) {    //错误处理
			cout << filename << "does not exist" << endl;
		}
		else {
			images.push_back(img);
		}
	}
	cout << images.size() << " images have been read." << endl;
	
	cout << "从图像中提取圆心" << endl;
	vector<vector<Point2f>> circle_center = extract_circle_center(images);

	return 0;
}
/*
 @param images: 要处理的图像
*/
vector<vector<Point2f>> extract_circle_center(vector<Mat> images) {
	vector<vector<Point2f>> circle_centers;//所有角点坐标 类型 Point2f
	vector<Point2f> ciircle_center;//单幅图片角点坐标
	for (Mat image : images) {
		//亚像素边缘求圆心
		Mat img;
		GaussianBlur(image, img, cv::Size(5, 5), 1.7);
		Canny(img, img, 0, 125);
		imshow("current", img);
		vector<vector<Point>> contours;
		findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //连通域提取，contours存储n个轮廓的边缘点信息

	}
	waitKey();
	destroyAllWindows();
	return circle_centers;
}