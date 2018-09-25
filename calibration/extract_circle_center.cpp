#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
using namespace cv;
vector<vector<Point2f>> extract_circle_center(vector<Mat> images);//��������,ʵ���ں��
vector<Point2f> contour_correct(Mat &image, vector<Point> contour);
bool iscontinous(const vector<Point>& contour);
float find_circle_center(vector<Point2f> & contour, Point center);//Բ����Ϻ���
void wate(vector<Point2f>& X, int show=0);
void ellipse_params(const Mat &u, Mat z, double a, double b, double alpha, double err, int show = 0);

int main() {
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
	vector<vector<Point2f>> circle_centers;		//���нǵ����� ���� Point2f	
	int cnt = 0;
	for (Mat image : images) {
		cout << "processing image No." << ++cnt << endl;
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
			if ((*contour).size()< 50|| (*contour).size()>350 ) {
				contour = contours.erase(contour);
			}
			else if (!iscontinous(*contour)) {
				contour = contours.erase(contour);
			}
			else {
				vector<Point2f> tmp=contour_correct(image, *contour);
				contours_subpixel.push_back(tmp);//��ȡ�����ؼ���Ե��Ϣ����push��contours_subpixel
				Point p;
				float rerr = find_circle_center(tmp, p);
				contour++;				
			}
		}
		Mat tmp(image.size(), CV_8U, Scalar(255));
		drawContours(tmp, contours, -1, Scalar(0), 1);
		imwrite("contour" + to_string(cnt)+".bmp", tmp);
	}
	return circle_centers;
}

/* �����ر�Ե��ȡ
** @param image: ��Ӧͼ��
** @param contour: �����ر�Ե��ļ���
** ĳ������Χ5*5С�棨Z������ϣ�����Ϻ������ݶȷ���Ͷ��׵�����������ر�Ե
*/
vector<Point2f> contour_correct(Mat &image, const vector<Point> contour) {
	vector<Point2f> points_subpixel;
	for (auto point : contour) {		
		int x = point.y, y = point.x;
		if (y > image.cols - 3 || y < 2) continue;
		if (x < 2 || x > image.rows - 3) continue;
		Mat Z = (image.rowRange(x - 2, x + 3).colRange(y - 2, y + 3).clone()).reshape(1, 25).clone();
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
	}
	return points_subpixel;
}

/*�жϱ�Ե�Ƿ�����
  �����ı�Ե���е�ֻ����һ��
  ʱ�临�Ӷ�n*n ���Ż�
*/
bool iscontinous(const vector<Point>& contour) {
	for (auto x : contour) {
		int cnt = 0;
		for (auto y : contour) {
			if (x==y) {
				cnt++;
				if (cnt > 2) {
					return false;
				}
			}
		}
	}
	return true;
}

float find_circle_center(vector<Point2f> & contour,Point center) {
	wate(contour);
	return 0.0;
}

void wate(vector<Point2f>& X,int show) {
	Mat coordinates = Mat(X);
	int delta = 1;
	int omega = 0;
	double myeps = 1e-6;
	Scalar mean_X = mean(coordinates);
	coordinates -= Mat(coordinates.size(), CV_32FC2, Scalar(mean_X[0],mean_X[1]));
	double  tmp=norm(coordinates,NORM_INF);
	coordinates /= tmp;
	vector<Mat> channels;
	split(coordinates, channels);
	Mat x1 = channels.at(0);
	Mat x2 = channels.at(1);
	Mat oldu = Mat::zeros(Size(1, 6), CV_32F);
	Mat w1 = Mat::ones(Size(1, X.size()), CV_32F);
	Mat A1;
	hconcat(x1.mul(x1), x1.mul(x2), A1);
	hconcat(A1, x2.mul(x2), A1);
	hconcat(A1, x1, A1);
	hconcat(A1, x2, A1);
	hconcat(A1, w1, A1);
	Mat A2 = (Mat_<float>(2,6) << 1,0,-1,0,0,0,0,1,0,0,0,0);
	int step = 20;
	while (step > 0) {
		Mat S, U, VT;
		Mat col_1 = Mat::diag(w1)*A1;
		Mat col_2 = omega * norm(w1, NORM_INF)*A2;
		vconcat(col_1, col_2, col_1);
		SVDecomp(col_1, S, U, VT, SVD::FULL_UV);
		Mat u = VT.row(5);
		transpose(u, u);
		if (norm(oldu - u) < myeps) {
			break;
		}
		double  a=0, b=0, alpha=0, err=0;
		Mat z;
		ellipse_params(u, z, a, b, alpha, err);
		step--;
	}

}



void ellipse_params(const Mat &u,Mat z,double a,double b,double alpha,double err, int show) {
	err = 0;
	vector<float> u_vec(u);
	Mat A = (Mat_<float>(2, 2) << u_vec[0], u_vec[1] / 2.0, u_vec[1] / 2.0, u_vec[2]);
	Mat B = (Mat_<float>(2, 1) << u_vec[3], u_vec[4]);
	B.convertTo(B, CV_32F);
	double C = u_vec[5];
	Mat val,vec;
	eigen(A, val,vec);
	double det = val.ptr<double>(0)[0] * val.ptr<double>(0)[1];
	if (det <= 0) {//��������Բ����
		err = 1;
		z = (Mat_<float>(1, 2) << 0, 0);
		a = 1; b = 1; alpha = 0;
	}
	else {
		alpha = atan2(vec.ptr<double>(1)[1], vec.ptr<double>(0)[1]);
		Mat vec_t,val_inv;
		transpose(vec, vec_t);
		Mat bs = vec_t * B;
		double k = invert(-2 * val, val_inv, DECOMP_SVD);
		Mat zs = val_inv * bs;
		z = zs.ptr<double>(0)[0] * vec;
		transpose(bs, bs);
		Mat h= -bs * zs.ptr<double>(0)[0] / 2 - C;
		a = sqrt(h.ptr<float>(0)[0] / val.ptr<float>(0)[0]);
		b = sqrt(h.ptr<float>(0)[0] / val.ptr<float>(0)[1]);
	}
}