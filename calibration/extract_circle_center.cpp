#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;

vector<vector<Point2f>> extract_circle_center(vector<Mat> images);//��������,ʵ���ں��

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
	glob(path, filenames);//��ָ��·���µ��ļ�������filenames����

	for (String filename:filenames) {  //���ζ���ָ��ͼƬ��������images������
		cout << "reading " << filename << endl;
		Mat img = imread(filename,0);
		if (img.empty()) {    //������
			cout << filename << "does not exist" << endl;
		}
		else {
			images.push_back(img);
		}
	}
	cout << images.size() << " images have been read." << endl;
	
	cout << "��ͼ������ȡԲ��" << endl;
	vector<vector<Point2f>> circle_center = extract_circle_center(images);

	return 0;
}
/*
 @param images: Ҫ�����ͼ��
*/
vector<vector<Point2f>> extract_circle_center(vector<Mat> images) {
	vector<vector<Point2f>> circle_centers;//���нǵ����� ���� Point2f
	vector<Point2f> ciircle_center;//����ͼƬ�ǵ�����
	for (Mat image : images) {
		//�����ر�Ե��Բ��
		Mat img;
		GaussianBlur(image, img, cv::Size(5, 5), 1.7);
		Canny(img, img, 0, 125);
		imshow("current", img);
		vector<vector<Point>> contours;
		findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //��ͨ����ȡ��contours�洢n�������ı�Ե����Ϣ

	}
	waitKey();
	destroyAllWindows();
	return circle_centers;
}