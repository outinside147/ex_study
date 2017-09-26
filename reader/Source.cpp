
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace cv;
using namespace std;


int main(int argc, const char* argv[])
{
	int hr = -1;
	try
	{
		Mat src,dst, dst2, edge;

		// �摜�ǂݍ���
		src = imread("../images/source/DSC01095.JPG", 0); //�O���[�X�P�[���摜�Ƃ��ēǂݍ���
		namedWindow("src", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("src", src);

		/// cornerHarris�̊e�l
		int blockSize = 3;			// Default = 2		,matched = 3	,minimum = 3
		int apertureSize = 7;		// Default = 3		,matched = 7	,minimum = 7
		double k = 0.2;				// Default = 0.04	,matched = 0.06	,minimum = 0.2 

		/// �n���X�̕��@�œ����_�����o����
		cornerHarris(src, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

		// �G�b�W�摜(�����_�摜)
		imwrite("../images/dst.png", dst);

		// 臒l 0.1 �� [ 0 - 1 ] �͈͓̔��łQ�l��
		// dst2..2�l���摜 , THRESH_BINARY..0.1�ȏ�ł����1,����ȊO�ł����0�Ƃ���B
		threshold(dst, dst2, 0.1, 1, THRESH_BINARY);
		imwrite("../images/dst2.png", dst2);
		// �̈�m��
		edge = Mat(src.rows, src.cols, CV_8UC3);

		// src[0] -> dst[2], src[0] -> dst[1], src[0] -> dst[0]
		// �Ƃ������ɁA�R�s�[���C���f�b�N�X�ƃR�s�[��C���f�b�N�X���w�肷��
		int from_to[] = { 0,2, 0,1, 0,0 };
		// �O���[�X�P�[�����R�`�����l���ɑ��₷
		mixChannels(&src, 1, &edge, 1, from_to, 3);
		
		// ���o���ꂽ�G�b�W�����ƂɐԂœh��Ԃ�
		int count = 0; //�����_�̃J�E���g
		Mat Ifeatures = Mat::zeros(src.rows, src.cols, CV_8UC1); // �����_�̏ꏊ��1�Ƃ���z��
		
		//ofstream ofs("../feature_points.txt"); //�e�L�X�g�ɏo��
		for (int i = 0; i < edge.rows; i++)   // ����
		{
			for (int j = 0; j < edge.cols; j++)   // ��
			{
				// .at���\�b�h..��f���s���|�C���g�œǂݍ���
				// 2�l���摜�̎w�肵���_��1�ł���ΐԂ��h��Ԃ�
				if (dst2.at< Vec<float, 1>>(i, j)[0] > 0)
				{
					edge.at< Vec3b>(i, j)[2] = 255;
					//�����_�Ȃ�΃t���O("1")����
					Ifeatures.at<unsigned char>(i, j) = 1;
					//ofs << "pt = " << pt << endl; //�e�L�X�g�ɏo��
					count++;
				}			
			}
		}
		cout << "count = " << count << endl;
		cout << "S.cols = " << Ifeatures.cols << " ,S.rows = " << Ifeatures.rows << endl;

		/* �c�����ŁA�s���E���獶�ɐi�ނ��Ƃ����肵������ */

		// �����_�ߖT���`����֐�
		int mgn1 = 75;
		int mgn2 = 25;
		int th = 5; //default=5
		// �؂�o���������Z�O�����g�̈ʒu�ƕ����i�[����ϐ�
		int num = 0;
		Mat segment = Mat::zeros(8000, 5, CV_32SC1);
		Mat opt_seg;

		/* �����_�摜���E���獶�ɁA�E�Ή������ɃX�L�������邱�Ƃŕ����Z�O�����g�̏����l���v�Z���� */

		// �摜�̐؂蔲��,�����̈�̔���
		int roop_cnt = 0;
		int match_cnt = 0;
		Mat col_vec, row_vec; //col_vec..��x�N�g��,row_vec..�s�x�N�g��
		for (int k = dst2.cols - mgn2; k >= 1 + mgn1; k--){
			int x = k;
			int y = 1 + mgn2;
			int xrng, yrng;
			while (1){
				if (Ifeatures.at<unsigned char>(y, x) == 1){
					match_cnt++;
					yrng = (y + mgn1) - (y - mgn2) + 1; //(2496-2396) + 1
					xrng = (x + mgn2) - (x - mgn1) + 1; //(882-782) + 1
					// �����_�摜��101*101�̗̈���摜�Ƃ��ĕۑ�
					Rect rect(x - mgn1, y + mgn1, xrng, yrng);
					Mat part_img(Ifeatures, rect);
					//imwrite("../images/rect/rect_" + std::to_string(k) + ".png", part_img);
					reduce(part_img, col_vec, 1, CV_REDUCE_SUM, CV_32F); // 101*101�̍s���1��ɏk��(�e��̍��v�l)
					reduce(col_vec, row_vec, 0, CV_REDUCE_SUM, CV_32F);  // 101*1�̍s���1�s�ɏk��(�e�s�̍��v�l)
					// �����_�ߖT�̓����_�̐���th�ȉ��ł���Ε����̈�ł͂Ȃ��Ɣ��肷��
					if (row_vec.at<int>(0, 0) <= th){
						//cout << "break_not_term" << endl;
						break;
					}
					// �����_�ߖT�̓����_�̐���th���������ꍇ
					Mat roi(src, rect);
					Mat roibw(roi.rows, roi.cols, CV_8UC1);
					Mat reduct_img(roi.rows, roi.cols, CV_8UC1);
					Mat ref_img;
					// ��Â̕��@��2�l��
					threshold(roi, roibw, 0, 1, THRESH_BINARY | THRESH_OTSU);
					// 3*3�̐����`�̍\�����v�f�Ŏ��k����
					erode(roibw, reduct_img, Mat(), cv::Point(-1, -1),1);
					// ���������ɓ��e����(�񂲂Ƃ̘a���v�Z)
					reduce(reduct_img,ref_img, 0, CV_REDUCE_SUM, CV_32F);

					//���e�f�[�^�ɋ��E�������flg=1,�Ȃ����flg=2
					//int rgt,lft,top,btm,flg;
					double rgt,lft,top,btm,flg;
					for (rgt = mgn1+2 ; rgt <= mgn1 + mgn2+1 ; rgt++){
						if (ref_img.at<float>(0,rgt-1) == roibw.rows){
							//cout << "break_rgt1" << endl;
							break;
						}
					}
					for (lft = mgn1; lft >= 1; lft--){
						if (ref_img.at<float>(0, lft-1) == roibw.rows){
							//cout << "break_lft1" << endl;
							break;
						}
					}
					if (rgt != mgn1 + mgn2 + 1 && lft != 1){
						flg = 1; //�����̈�Ɣ���
					} else {
						flg = 2; //�����̈�ł͂Ȃ��Ɣ���
					}
					
					//�摜��̍��W�ɕϊ�����
					rgt = rgt + x - (mgn1 + 1);
					lft = lft + x - (mgn1 + 1);
					top = y - mgn2;
					btm = y + mgn1;
					
					if(flg == 2){
						// �����̈�łȂ��͈͂̓����_������
						for (int i = top-1; i < btm; i++){
							for (int j = lft-1; j < rgt; j++){
								if (dst2.at< Vec<float, 1>>(i, j)[0] > 0){
									Ifeatures.at<unsigned char>(i, j) = 0;
									edge.at< Vec3b>(i, j)[2] = 0;
								}
							}
						}
						//cout << "break_delete_Ifeatures" << endl;
						break;
					}

					// �������������_������
					for (int i = top-1; i < btm; i++){
						for (int j = lft-1; j < rgt; j++){
							if (dst2.at< Vec<float, 1>>(i, j)[0] > 0){
								Ifeatures.at<unsigned char>(i, j) = 0;
								edge.at< Vec3b>(i, j)[2] = 0;
							}
						}
					}
					double xctr = round((lft + rgt)/2);

					// segment��num�Ԗڂ̍s�ɒ��o���������̈���i�[
					segment.at<int>(num, 0) = 1; //�����̈�̍��W���L�^
					segment.at<int>(num, 1) = top;
					segment.at<int>(num, 2) = btm;
					segment.at<int>(num, 3) = lft;
					segment.at<int>(num, 4) = rgt;
					num++;
					//cout << "seg_num1=" << num << endl;

					/* �����̈�Ɣ��肵���ꏊ���牺�����ɕ����̈��T������ */
					int yy = y + mgn1 + mgn2 + 1;
					while (1){
						// �摜�̉��[�ɒB����ΏI��
						if (yy > src.rows - mgn1){
							//cout << "break_limit" << endl;
							break;
						}
						// �����Ώۂ̋ߖT�̈��ݒ肷��
						yrng = (yy + mgn1) - (yy - mgn2) + 1;
						xrng = (xctr + mgn2) - (xctr - mgn2) + 1;
						Rect rect(x - mgn1, y + mgn1, xrng, yrng);
						Mat part_img(edge, rect);
						reduce(part_img, col_vec, 1, CV_REDUCE_SUM, CV_32F); // 100*100�̍s���1��ɏk��(�e��̍��v�l)
						reduce(col_vec, row_vec, 0, CV_REDUCE_SUM, CV_32F);  // 100*1�̍s���1�s�ɏk��(�e�s�̍��v�l)
						// ���̒��ɓ����_���Ȃ���΂��̗�̒T�����I������
						if (row_vec.at<int>(0, 0) == 0) break;
						// �ߖT�̈���l���A���k���A���e����
						Mat roi(src, rect);
						Mat roibw(roi.rows, roi.cols, CV_8UC1);
						Mat reduct_img(roi.rows, roi.cols, CV_8UC1);
						Mat ref_img,ref_img2;
						threshold(roi, roibw, 0, 1, THRESH_BINARY | THRESH_OTSU);
						erode(roibw, reduct_img, Mat(), cv::Point(-1, -1), 1);
						reduce(reduct_img, ref_img, 0, CV_REDUCE_SUM, CV_32F); // 1*101�s��
						reduce(reduct_img, ref_img2, 1, CV_REDUCE_SUM, CV_32F); // 101*1�s��
						// ���e�f�[�^���當�������݂���͈͂𐄒肷��
						for (rgt = mgn2 + 2; rgt <= mgn2 + mgn2 + 1; rgt++){
							if (ref_img.at<float>(0, rgt-1) == roibw.rows){
								//cout << "break_rgt2" << endl;
								break;
							}
						}
						for (lft = mgn2; lft >= 1; lft--){
							if (ref_img.at<float>(0, lft-1) == roibw.rows){
								//cout << "break_lft2" << endl;
								break;
							}
						}
						for (btm = mgn1 + mgn2 + 1; btm >= 1; btm--){
							if (ref_img2.at<float>(btm-1, 0) != roibw.cols){
								//cout << "break_btm2" << endl;
								break;
							}
						}
						//�摜��̍��W�ɕϊ�����
						rgt = rgt + xctr - (mgn2 + 1);
						lft = lft + xctr - (mgn2 + 1);
						top = yy - mgn2;
						btm = btm + yy - (mgn2 + 1);
						
						// �������������_������
						for (int i = top-1; i < btm; i++){
							for (int j = lft-1; j < rgt; j++){
								if (dst2.at< Vec<float, 1>>(i, j)[0] > 0){
									Ifeatures.at<unsigned char>(i, j) = 0;
									edge.at< Vec3b>(i, j)[2] = 0;
								}
							}
						}
						
						// segment��num�Ԗڂ̍s�ɒ��o���������̈���i�[
						segment.at<int>(num, 0) = 2; //�����̈�ł͂Ȃ����W���L�^
						segment.at<int>(num, 1) = top;
						segment.at<int>(num, 2) = btm;
						segment.at<int>(num, 3) = lft;
						segment.at<int>(num, 4) = rgt;
						num++;
						//cout << "seg_num2=" << num << endl;

						//��������̈�����Ɉړ�����
						xctr = round((lft + rgt) / 2);
						yy = btm + 1 + mgn2;
					}
					//cout << "break_roop" << endl;
					break;
				}

				//�E���̉�f�Ɉړ�����
				x = x + 1;
				y = y + 1;
				if (x > dst2.cols - mgn2 || y > dst2.rows - mgn1){
					//cout << "break_over_end" << endl;
					break;
				}
			}
		}

		cout << "num=" << num << endl;

		//�z��̑傫�����œK������(�l��0�łȂ��s�܂ł��c��)
		opt_seg = Mat::zeros(num, 5, CV_32SC1);
		opt_seg = segment.rowRange(cv::Range(0, num));	//�Z�O�����g��0�`num�s�ڂ܂ł�ϐ��ɑ��
		cout << "opt_seg.size=" << opt_seg.size() << endl;

		/////////////////////////////////////////////////////////////////////////////

		/* ���o���������Z�O�����g�𐮌`���� */

		// ��̐�[�̈ʒu�𒼐��ߎ����A��������傫���������̂����O����B
		// ���[�̃Z�O�����g���𐔂���		
		int tnum = 0;
		for (int i = 0; i < segment.rows; i++){
			if (segment.at<int>(i, 0) == 1){ //i�s0��
				tnum++;
			}
		}
		cout << "tnum=" << tnum << endl; // tnum=55

		// ���[�̃Z�O�����g�̔ԍ��ƍ��W��topseg�ɃR�s�[����		
		Mat topseg = Mat::zeros(tnum, 3, CV_32SC1); // tnum�s3��
		tnum = 0;
		for (int i = 0; i < segment.rows ; i++){
			if (segment.at<int>(i, 0) == 1){
				topseg.at<int>(tnum, 0) = i;
				topseg.at<int>(tnum, 1) = segment.at<int>(i, 3); // lft
				topseg.at<int>(tnum, 2) = segment.at<int>(i, 1); // top
				tnum++;
			}		
		}

		// ���[�̍��W����ARANSAC�Œ����𐄒肵�A���肵�������Ɨ��[�Z�O�����g�̋��������߂�
		int lnum = 10;
		int rn[2] = {};
		//���肵�������Ɨ��[�Z�O�����g�̋������i�[����z��,double�^���i�[����悤�w��
		Mat seg_dst = Mat::zeros(lnum, tnum,CV_64FC1);
		for (int k = 0; k < lnum; k++){ //2�̗����𐶐�
			for (int j = 0; j < 2; j++){
				rn[j] = rand() % tnum;
			}
			if (rn[0] == rn[1]){ //2�̗������قȂ鐔�l�ɂȂ�悤�ɂ���
				rn[1] = rand() % tnum;
				while (rn[1] == rn[0]){
					rn[1] = rand() % tnum;
				}
			}

			//2�_�̍��W2~3���(C�̏ꍇ,1~2���)�����o��
			Mat xy1 = Mat::zeros(1, 2, CV_64FC1); //int�^
			Mat xy2 = xy1.clone();

			Mat v1 = Mat::zeros(2, 1, CV_64FC1); //double�^
			Mat v2 = Mat::zeros(2, 1, CV_64FC1);

			xy1.at<double>(0, 0) = topseg.at<int>(rn[0], 1);
			xy1.at<double>(0, 1) = topseg.at<int>(rn[0], 2);
			xy2.at<double>(0, 0) = topseg.at<int>(rn[1], 1);
			xy2.at<double>(0, 1) = topseg.at<int>(rn[1], 2);

			// xy1,xy2��]�u 1*2�s��2*1�s��֕ύX
			xy1 = xy1.t();
			xy2 = xy2.t();

			v1 = xy2 - xy1;

			// v1/norm(v1); 2�_�̍��W�����̒P�ʃx�N�g��
			// norm .. �x�N�g��v1�̃��[�N���b�h������Ԃ�
			// ���[�N���b�h���� .. v1(a,b) = ��(a^2 + b^2)
			v1 = v1 / norm(v1);

			// v2 = [v1[1] ; -v1[0]]; ��������P�ʃx�N�g��,����ւ����x�N�g��
			v2.at<double>(0, 0) = v1.at<double>(1, 0);
			v2.at<double>(1, 0) = -v1.at<double>(0, 0);

			// 2�_���猈�܂钼���Ƃ̋������v�Z����
			for (int i = 0; i < tnum; i++){
				Mat xy3 = Mat::zeros(1, 2, CV_64FC1);
				if (i == rn[0] || i == rn[1]){
					seg_dst.at<double>(k, tnum-1) = 0;
				} else {
					xy3.at<double>(0, 0) = topseg.at<int>(i, 1);
					xy3.at<double>(0, 1) = topseg.at<int>(i, 2);

					// xy3��]�u
					xy3 = xy3.t();

					// ab = [v2,-v1]\(xy1-xy3) = inv([v2,-v1])*(xy1-xy3)
					Mat a = Mat::zeros(2, 2, CV_64FC1);
					a.at<double>(0, 0) = v2.at<double>(0, 0);
					a.at<double>(1, 0) = v2.at<double>(1, 0);
					a.at<double>(0, 1) = -v1.at<double>(0, 0);
					a.at<double>(1, 1) = -v1.at<double>(1, 0);

					Mat ab = Mat::zeros(2, 1, CV_64FC1);
					//�t�s������߂�
					// a.inv() .. a�̋t�s��
					ab = a.inv() * (xy1 - xy3);
					seg_dst.at<double>(k, i) = fabs(ab.at<double>(0,0)); //fabs .. ����x�̐�Βl���v�Z���A���ʂ�double�^�ŕԂ�
				}
			}
		}

		// lnum�{�̒����̒���,tnum�̋����l�̒����l���ŏ��̂��̂��ł��ǂ��ߎ������Ƃ���
		// �����l�����߂�
		Mat dst_sort;
		// �e�s��傫���̏��ɕ��ёւ���		
		cv::sort(seg_dst, dst_sort, CV_SORT_EVERY_ROW|CV_SORT_ASCENDING); //�s��̊e�s�������Ƀ\�[�g����
		//cout << "dst_sort=" << endl << dst_sort << endl;

		double med[10] = {};
		for (int h = 0; h < lnum; h++){ // lnum=10
			if (tnum % 2 == 1){ //�f�[�^��(��)����̏ꍇ
				med[h] = dst_sort.at<double>(h, (tnum-1)/2);
			}
			else { //�f�[�^��(��)�������̏ꍇ
				med[h] = (dst_sort.at<double>(h, (tnum / 2) - 1) + dst_sort.at<double>(h, tnum / 2)) / 2.0;
			}
			cout << h << ".med= " << med[h] << endl;
		}

		//�ŏ��l�����߂�
		double min = 9999; //�ŏ��l��������ϐ�
		int index = 0;
		for (int i = 0; i < lnum; i++){
			if (med[i] < min){
				min = med[i];
				index = i;			
			}
		}
		cout << "min= " << min << ",index= " << index << endl; //�ŏ��l�Ƃ��̏ꏊ

		// �ߎ������܂ł̋�����th�ȏ�̗��[�Z�O�����g�Ƃ���ɑ����Z�O�����g����������
		th = 150; //default=150
		for (int i = 0; i < tnum; i++){
			if (seg_dst.at<double>(index, i) > th){
				k = topseg.at<int>(i, 0);
				segment.at<int>(k, 0) = 0;
				k++;
				while (k <= segment.rows && segment.at<int>(k, 0) == 2){
					segment.at<int>(k, 0) = 0;
					k++;
				}
			}
		}

		for (int i = 0; i < num; i++){
			if (segment.at<int>(i, 0) != 0){
				int top = segment.at<int>(i, 1);
				int btm = segment.at<int>(i, 2);
				int lft = segment.at<int>(i, 3);
				int rgt = segment.at<int>(i, 4);
				int rect_width = 0;
				rect_width = abs(lft - rgt);
				//if (rect_width >= 10){ //���������܂�ɂ����������̂͏��O
					cv::rectangle(edge, cv::Point(lft, top), cv::Point(rgt, btm), cv::Scalar(0, 0, 200), 3, 4);
				//}
			}
		}

		/* ���ʉ摜�̕\�� */
		namedWindow("edge", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("edge", edge);
		imwrite("../images/edge.png", edge);
		waitKey(0);
		hr = 0;
	}
	catch (Exception ex)
	{
		cout << ex.err << endl;
	}
	// �E�B���h�E�̔j��
	destroyAllWindows();
	return hr;
}