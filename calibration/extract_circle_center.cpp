#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
vector<vector<Point2f>> extract_circle_center(vector<Mat> images);//��������,ʵ���ں��
vector<Point2f> contour_correct(Mat image, vector<Point> contour);

int main() {
	/*��ʽ��ʼ*/
	string prefix;
	string type;
	vector<String> filenames;
	vector<Mat> images;
	cout << "\tspecify the image prefix" << endl;
	cin >> prefix;
	cout << "\tspecify the image type" << endl;
	cin >> type;
	string path = ".\\image\\" + prefix + "*." + type;
	glob(path, filenames);							//��ָ��·���µ��ļ�������filenames����

	for (String filename : filenames) {				//���ζ���ָ��ͼƬ��������images������
		cout << "\treading " << filename << endl;
		Mat img = imread(filename, 0);
		if (img.empty()) {							//������
			cout << "\t" << filename << "does not exist" << endl;
		}
		else {
			images.push_back(img);
		}
	}
	cout << "\t" << images.size() << " images have been read." << endl;

	cout << "\t��ͼ������ȡԲ��" << endl;

	vector<vector<Point2f>> circle_center = extract_circle_center(images);

	return 0;
}
/*
@param images: Ҫ�����ͼ��
*/
vector<vector<Point2f>> extract_circle_center(vector<Mat> images) {
	vector<vector<Point2f>> circle_centers;					//���нǵ����� ���� Point2f	
	for (Mat image : images) {
		vector<Point2f> circle_center;						//����ͼƬԲ������
		Mat img(image.size(), CV_8U, Scalar(255));
		vector<vector<Point>> contours;						//�����ر�Ե
		vector<vector<Point2f>> contours_subpixel;			//�洢�����ر�Ե
		GaussianBlur(image, image, Size(5, 5), 1.7);
		Canny(image, img, 0, 125);
		Mat kernel(3, 3, CV_8U, Scalar(1));
		morphologyEx(img, img, MORPH_CLOSE, kernel);
		findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		/*ȥ��̫���Լ�̫С�ı�Ե����Сͳ�Ʒ�Χ*/
		for (auto contour = contours.begin(); contour != contours.end();) {
			if (contourArea(*contour) < 200) {
				contour = contours.erase(contour);
			}
			else {
				contours_subpixel.push_back(contour_correct(image, *contour));//��ȡ�����ؼ���Ե��Ϣ����push��contours_subpixel
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
		Z.convertTo(Z, CV_32FC1);//ͳһ��������,��ֻ���� CV_32FC1 CV_32FC2 CV_64FC1 CV_64FC2
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
		K = A_inv * Z;//����˷�Ҫ��֤��������һ��
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
		Point2f  corr = point;//����������
		corr.x += r * cos_theta;
		corr.y += r * sin_theta;
		points_subpixel.push_back(corr);
		cout << corr << endl;
	}
	return points_subpixel;
}