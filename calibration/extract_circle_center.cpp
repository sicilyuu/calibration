#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
vector<vector<Point2f>> extract_circle_center(vector<Mat> images);//函数声明,实体在后边
vector<Point2f> contour_correct(Mat image, vector<Point> contour);

int main() {
	/*正式开始*/
	string prefix;
	string type;
	vector<String> filenames;
	vector<Mat> images;
	cout << "\tspecify the image prefix" << endl;
	cin >> prefix;
	cout << "\tspecify the image type" << endl;
	cin >> type;
	string path = ".\\image\\" + prefix + "*." + type;
	glob(path, filenames);							//将指定路径下的文件名存入filenames向量

	for (String filename : filenames) {				//依次读入指定图片，并存入images向量中
		cout << "\treading " << filename << endl;
		Mat img = imread(filename, 0);
		if (img.empty()) {							//错误处理
			cout << "\t" << filename << "does not exist" << endl;
		}
		else {
			images.push_back(img);
		}
	}
	cout << "\t" << images.size() << " images have been read." << endl;

	cout << "\t从图像中提取圆心" << endl;

	vector<vector<Point2f>> circle_center = extract_circle_center(images);

	return 0;
}
/*
@param images: 要处理的图像
*/
vector<vector<Point2f>> extract_circle_center(vector<Mat> images) {
	vector<vector<Point2f>> circle_centers;					//所有角点坐标 类型 Point2f	
	for (Mat image : images) {
		vector<Point2f> circle_center;						//单幅图片圆心坐标
		Mat img(image.size(), CV_8U, Scalar(255));
		vector<vector<Point>> contours;						//整像素边缘
		vector<vector<Point2f>> contours_subpixel;			//存储亚像素边缘
		GaussianBlur(image, image, Size(5, 5), 1.7);
		Canny(image, img, 0, 125);
		Mat kernel(3, 3, CV_8U, Scalar(1));
		morphologyEx(img, img, MORPH_CLOSE, kernel);
		findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		/*去掉太大以及太小的边缘，缩小统计范围*/
		for (auto contour = contours.begin(); contour != contours.end();) {
			if (contourArea(*contour) < 200) {
				contour = contours.erase(contour);
			}
			else {
				contours_subpixel.push_back(contour_correct(image, *contour));//获取亚像素级边缘信息，并push进contours_subpixel
				contour++;
			}
		}
		Mat tmp(image.size(), CV_8U, Scalar(255));
		drawContours(tmp, contours, -1, Scalar(0), 1);
		imshow("img", img);
		imshow("contour", tmp);
	}
	waitKey();
	destroyAllWindows();
	return circle_centers;
}

vector<Point2f> contour_correct(Mat image, const vector<Point> contour) {
	vector<Point2f> points_subpixel;
	for (auto point : contour) {
		int x = point.x, y = point.y;
		Mat Z = (image.rowRange(x - 2, x + 3).colRange(x - 2, x + 3).clone()).reshape(1, 25).clone();
		Mat X = (Mat_<float>(5, 5) << -2, -1, 0, 1, 2,
			-2, -1, 0, 1, 2,
			-2, -1, 0, 1, 2,
			-2, -1, 0, 1, 2,
			-2, -1, 0, 1, 2);
		Mat Y, K, A_inv;
		Mat A = Mat::ones(25, 1, CV_32FC1);
		transpose(X, Y);
		X = X.reshape(1, 25).clone();
		Y = Y.reshape(1, 25).clone();
		Z.convertTo(Z, CV_32FC1);//统一数据类型,且只能是 CV_32FC1 CV_32FC2 CV_64FC1 CV_64FC2
		X.convertTo(X, CV_32FC1);
		Y.convertTo(Y, CV_32FC1);
		hconcat(A, Y, A);
		hconcat(A, X, A);
		hconcat(A, Y.mul(Y), A);
		hconcat(A, X.mul(Y), A);
		hconcat(A, X.mul(X), A);
		hconcat(A, Y.mul(Y).mul(Y), A);
		hconcat(A, X.mul(Y).mul(Y), A);
		hconcat(A, X.mul(X).mul(Y), A);
		hconcat(A, X.mul(X).mul(X), A);
		double k = invert(A, A_inv, DECOMP_SVD);
		K = A_inv * Z;//矩阵乘法要保证数据类型一致
		float k2 = K.ptr<float>(1)[0];
		float k3 = K.ptr<float>(2)[0];
		float k4 = K.ptr<float>(3)[0];
		float k5 = K.ptr<float>(4)[0];
		float k6 = K.ptr<float>(5)[0];
		float k7 = K.ptr<float>(6)[0];
		float k8 = K.ptr<float>(7)[0];
		float k9 = K.ptr<float>(8)[0];
		float k10 = K.ptr<float>(9)[0];
		float temp = sqrt(k2*k2 + k3 * k3);
		float sin_theta = k2 / temp;
		float cos_theta = k3 / temp;
		float a = 6 * (k7*sin_theta*sin_theta*sin_theta + k8 * sin_theta*sin_theta*cos_theta + k9 * sin_theta*cos_theta*cos_theta + k10 * cos_theta*cos_theta*cos_theta);
		float b = 2 * (k4*sin_theta*sin_theta + k5 * sin_theta*cos_theta + k6 * cos_theta*cos_theta);
		float r = (-b) / a;
		Point2f  corr = point;//亚像素修正
		corr.x += r * cos_theta;
		corr.y += r * sin_theta;
		points_subpixel.push_back(corr);
		cout << corr << endl;
	}
	return points_subpixel;
}