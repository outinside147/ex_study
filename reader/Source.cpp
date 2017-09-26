
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

		// 画像読み込み
		src = imread("../images/source/DSC01095.JPG", 0); //グレースケール画像として読み込み
		namedWindow("src", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("src", src);

		/// cornerHarrisの各値
		int blockSize = 3;			// Default = 2		,matched = 3	,minimum = 3
		int apertureSize = 7;		// Default = 3		,matched = 7	,minimum = 7
		double k = 0.2;				// Default = 0.04	,matched = 0.06	,minimum = 0.2 

		/// ハリスの方法で特徴点を検出する
		cornerHarris(src, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

		// エッジ画像(特徴点画像)
		imwrite("../images/dst.png", dst);

		// 閾値 0.1 で [ 0 - 1 ] の範囲内で２値化
		// dst2..2値化画像 , THRESH_BINARY..0.1以上であれば1,それ以外であれば0とする。
		threshold(dst, dst2, 0.1, 1, THRESH_BINARY);
		imwrite("../images/dst2.png", dst2);
		// 領域確保
		edge = Mat(src.rows, src.cols, CV_8UC3);

		// src[0] -> dst[2], src[0] -> dst[1], src[0] -> dst[0]
		// という風に、コピー元インデックスとコピー先インデックスを指定する
		int from_to[] = { 0,2, 0,1, 0,0 };
		// グレースケールを３チャンネルに増やす
		mixChannels(&src, 1, &edge, 1, from_to, 3);
		
		// 検出されたエッジをもとに赤で塗りつぶす
		int count = 0; //特徴点のカウント
		Mat Ifeatures = Mat::zeros(src.rows, src.cols, CV_8UC1); // 特徴点の場所を1とする配列
		
		//ofstream ofs("../feature_points.txt"); //テキストに出力
		for (int i = 0; i < edge.rows; i++)   // 高さ
		{
			for (int j = 0; j < edge.cols; j++)   // 幅
			{
				// .atメソッド..画素をピンポイントで読み込む
				// 2値化画像の指定した点が1であれば赤く塗りつぶす
				if (dst2.at< Vec<float, 1>>(i, j)[0] > 0)
				{
					edge.at< Vec3b>(i, j)[2] = 255;
					//特徴点ならばフラグ("1")を代入
					Ifeatures.at<unsigned char>(i, j) = 1;
					//ofs << "pt = " << pt << endl; //テキストに出力
					count++;
				}			
			}
		}
		cout << "count = " << count << endl;
		cout << "S.cols = " << Ifeatures.cols << " ,S.rows = " << Ifeatures.rows << endl;

		/* 縦書きで、行が右から左に進むことを仮定した処理 */

		// 特徴点近傍を定義する関数
		int mgn1 = 75;
		int mgn2 = 25;
		int th = 5; //default=5
		// 切り出した文字セグメントの位置と幅を格納する変数
		int num = 0;
		Mat segment = Mat::zeros(8000, 5, CV_32SC1);
		Mat opt_seg;

		/* 特徴点画像を右から左に、右斜下方向にスキャンすることで文字セグメントの初期値を計算する */

		// 画像の切り抜き,文字領域の判定
		int roop_cnt = 0;
		int match_cnt = 0;
		Mat col_vec, row_vec; //col_vec..列ベクトル,row_vec..行ベクトル
		for (int k = dst2.cols - mgn2; k >= 1 + mgn1; k--){
			int x = k;
			int y = 1 + mgn2;
			int xrng, yrng;
			while (1){
				if (Ifeatures.at<unsigned char>(y, x) == 1){
					match_cnt++;
					yrng = (y + mgn1) - (y - mgn2) + 1; //(2496-2396) + 1
					xrng = (x + mgn2) - (x - mgn1) + 1; //(882-782) + 1
					// 特徴点画像の101*101の領域を画像として保存
					Rect rect(x - mgn1, y + mgn1, xrng, yrng);
					Mat part_img(Ifeatures, rect);
					//imwrite("../images/rect/rect_" + std::to_string(k) + ".png", part_img);
					reduce(part_img, col_vec, 1, CV_REDUCE_SUM, CV_32F); // 101*101の行列を1列に縮小(各列の合計値)
					reduce(col_vec, row_vec, 0, CV_REDUCE_SUM, CV_32F);  // 101*1の行列を1行に縮小(各行の合計値)
					// 特徴点近傍の特徴点の数がth個以下であれば文字領域ではないと判定する
					if (row_vec.at<int>(0, 0) <= th){
						//cout << "break_not_term" << endl;
						break;
					}
					// 特徴点近傍の特徴点の数がth個よりも多い場合
					Mat roi(src, rect);
					Mat roibw(roi.rows, roi.cols, CV_8UC1);
					Mat reduct_img(roi.rows, roi.cols, CV_8UC1);
					Mat ref_img;
					// 大津の方法で2値化
					threshold(roi, roibw, 0, 1, THRESH_BINARY | THRESH_OTSU);
					// 3*3の正方形の構造化要素で収縮処理
					erode(roibw, reduct_img, Mat(), cv::Point(-1, -1),1);
					// 垂直方向に投影する(列ごとの和を計算)
					reduce(reduct_img,ref_img, 0, CV_REDUCE_SUM, CV_32F);

					//投影データに境界があればflg=1,なければflg=2
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
						flg = 1; //文字領域と判定
					} else {
						flg = 2; //文字領域ではないと判定
					}
					
					//画像上の座標に変換する
					rgt = rgt + x - (mgn1 + 1);
					lft = lft + x - (mgn1 + 1);
					top = y - mgn2;
					btm = y + mgn1;
					
					if(flg == 2){
						// 文字領域でない範囲の特徴点を消去
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

					// 処理した特徴点を消去
					for (int i = top-1; i < btm; i++){
						for (int j = lft-1; j < rgt; j++){
							if (dst2.at< Vec<float, 1>>(i, j)[0] > 0){
								Ifeatures.at<unsigned char>(i, j) = 0;
								edge.at< Vec3b>(i, j)[2] = 0;
							}
						}
					}
					double xctr = round((lft + rgt)/2);

					// segmentのnum番目の行に抽出した文字領域を格納
					segment.at<int>(num, 0) = 1; //文字領域の座標を記録
					segment.at<int>(num, 1) = top;
					segment.at<int>(num, 2) = btm;
					segment.at<int>(num, 3) = lft;
					segment.at<int>(num, 4) = rgt;
					num++;
					//cout << "seg_num1=" << num << endl;

					/* 文字領域と判定した場所から下方向に文字領域を探索する */
					int yy = y + mgn1 + mgn2 + 1;
					while (1){
						// 画像の下端に達すれば終了
						if (yy > src.rows - mgn1){
							//cout << "break_limit" << endl;
							break;
						}
						// 処理対象の近傍領域を設定する
						yrng = (yy + mgn1) - (yy - mgn2) + 1;
						xrng = (xctr + mgn2) - (xctr - mgn2) + 1;
						Rect rect(x - mgn1, y + mgn1, xrng, yrng);
						Mat part_img(edge, rect);
						reduce(part_img, col_vec, 1, CV_REDUCE_SUM, CV_32F); // 100*100の行列を1列に縮小(各列の合計値)
						reduce(col_vec, row_vec, 0, CV_REDUCE_SUM, CV_32F);  // 100*1の行列を1行に縮小(各行の合計値)
						// その中に特徴点がなければその列の探索を終了する
						if (row_vec.at<int>(0, 0) == 0) break;
						// 近傍領域を二値化、収縮し、投影する
						Mat roi(src, rect);
						Mat roibw(roi.rows, roi.cols, CV_8UC1);
						Mat reduct_img(roi.rows, roi.cols, CV_8UC1);
						Mat ref_img,ref_img2;
						threshold(roi, roibw, 0, 1, THRESH_BINARY | THRESH_OTSU);
						erode(roibw, reduct_img, Mat(), cv::Point(-1, -1), 1);
						reduce(reduct_img, ref_img, 0, CV_REDUCE_SUM, CV_32F); // 1*101行列
						reduce(reduct_img, ref_img2, 1, CV_REDUCE_SUM, CV_32F); // 101*1行列
						// 投影データから文字が存在する範囲を推定する
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
						//画像上の座標に変換する
						rgt = rgt + xctr - (mgn2 + 1);
						lft = lft + xctr - (mgn2 + 1);
						top = yy - mgn2;
						btm = btm + yy - (mgn2 + 1);
						
						// 処理した特徴点を消去
						for (int i = top-1; i < btm; i++){
							for (int j = lft-1; j < rgt; j++){
								if (dst2.at< Vec<float, 1>>(i, j)[0] > 0){
									Ifeatures.at<unsigned char>(i, j) = 0;
									edge.at< Vec3b>(i, j)[2] = 0;
								}
							}
						}
						
						// segmentのnum番目の行に抽出した文字領域を格納
						segment.at<int>(num, 0) = 2; //文字領域ではない座標を記録
						segment.at<int>(num, 1) = top;
						segment.at<int>(num, 2) = btm;
						segment.at<int>(num, 3) = lft;
						segment.at<int>(num, 4) = rgt;
						num++;
						//cout << "seg_num2=" << num << endl;

						//処理する領域を下に移動する
						xctr = round((lft + rgt) / 2);
						yy = btm + 1 + mgn2;
					}
					//cout << "break_roop" << endl;
					break;
				}

				//右下の画素に移動する
				x = x + 1;
				y = y + 1;
				if (x > dst2.cols - mgn2 || y > dst2.rows - mgn1){
					//cout << "break_over_end" << endl;
					break;
				}
			}
		}

		cout << "num=" << num << endl;

		//配列の大きさを最適化する(値が0でない行までを残す)
		opt_seg = Mat::zeros(num, 5, CV_32SC1);
		opt_seg = segment.rowRange(cv::Range(0, num));	//セグメントの0～num行目までを変数に代入
		cout << "opt_seg.size=" << opt_seg.size() << endl;

		/////////////////////////////////////////////////////////////////////////////

		/* 抽出した文字セグメントを整形する */

		// 列の先端の位置を直線近似し、直線から大きく離れるものを除外する。
		// 列先端のセグメント数を数える		
		int tnum = 0;
		for (int i = 0; i < segment.rows; i++){
			if (segment.at<int>(i, 0) == 1){ //i行0列
				tnum++;
			}
		}
		cout << "tnum=" << tnum << endl; // tnum=55

		// 列先端のセグメントの番号と座標をtopsegにコピーする		
		Mat topseg = Mat::zeros(tnum, 3, CV_32SC1); // tnum行3列
		tnum = 0;
		for (int i = 0; i < segment.rows ; i++){
			if (segment.at<int>(i, 0) == 1){
				topseg.at<int>(tnum, 0) = i;
				topseg.at<int>(tnum, 1) = segment.at<int>(i, 3); // lft
				topseg.at<int>(tnum, 2) = segment.at<int>(i, 1); // top
				tnum++;
			}		
		}

		// 列先端の座標から、RANSACで直線を推定し、推定した直線と列先端セグメントの距離を求める
		int lnum = 10;
		int rn[2] = {};
		//推定した直線と列先端セグメントの距離を格納する配列,double型を格納するよう指定
		Mat seg_dst = Mat::zeros(lnum, tnum,CV_64FC1);
		for (int k = 0; k < lnum; k++){ //2つの乱数を生成
			for (int j = 0; j < 2; j++){
				rn[j] = rand() % tnum;
			}
			if (rn[0] == rn[1]){ //2つの乱数が異なる数値になるようにする
				rn[1] = rand() % tnum;
				while (rn[1] == rn[0]){
					rn[1] = rand() % tnum;
				}
			}

			//2点の座標2~3列目(Cの場合,1~2列目)を取り出す
			Mat xy1 = Mat::zeros(1, 2, CV_64FC1); //int型
			Mat xy2 = xy1.clone();

			Mat v1 = Mat::zeros(2, 1, CV_64FC1); //double型
			Mat v2 = Mat::zeros(2, 1, CV_64FC1);

			xy1.at<double>(0, 0) = topseg.at<int>(rn[0], 1);
			xy1.at<double>(0, 1) = topseg.at<int>(rn[0], 2);
			xy2.at<double>(0, 0) = topseg.at<int>(rn[1], 1);
			xy2.at<double>(0, 1) = topseg.at<int>(rn[1], 2);

			// xy1,xy2を転置 1*2行列→2*1行列へ変更
			xy1 = xy1.t();
			xy2 = xy2.t();

			v1 = xy2 - xy1;

			// v1/norm(v1); 2点の座標方向の単位ベクトル
			// norm .. ベクトルv1のユークリッド距離を返す
			// ユークリッド距離 .. v1(a,b) = √(a^2 + b^2)
			v1 = v1 / norm(v1);

			// v2 = [v1[1] ; -v1[0]]; 直交する単位ベクトル,入れ替えたベクトル
			v2.at<double>(0, 0) = v1.at<double>(1, 0);
			v2.at<double>(1, 0) = -v1.at<double>(0, 0);

			// 2点から決まる直線との距離を計算する
			for (int i = 0; i < tnum; i++){
				Mat xy3 = Mat::zeros(1, 2, CV_64FC1);
				if (i == rn[0] || i == rn[1]){
					seg_dst.at<double>(k, tnum-1) = 0;
				} else {
					xy3.at<double>(0, 0) = topseg.at<int>(i, 1);
					xy3.at<double>(0, 1) = topseg.at<int>(i, 2);

					// xy3を転置
					xy3 = xy3.t();

					// ab = [v2,-v1]\(xy1-xy3) = inv([v2,-v1])*(xy1-xy3)
					Mat a = Mat::zeros(2, 2, CV_64FC1);
					a.at<double>(0, 0) = v2.at<double>(0, 0);
					a.at<double>(1, 0) = v2.at<double>(1, 0);
					a.at<double>(0, 1) = -v1.at<double>(0, 0);
					a.at<double>(1, 1) = -v1.at<double>(1, 0);

					Mat ab = Mat::zeros(2, 1, CV_64FC1);
					//逆行列を求める
					// a.inv() .. aの逆行列
					ab = a.inv() * (xy1 - xy3);
					seg_dst.at<double>(k, i) = fabs(ab.at<double>(0,0)); //fabs .. 引数xの絶対値を計算し、結果をdouble型で返す
				}
			}
		}

		// lnum本の直線の中で,tnum個の距離値の中央値が最小のものを最も良い近似直線とする
		// 中央値を求める
		Mat dst_sort;
		// 各行を大きさの順に並び替える		
		cv::sort(seg_dst, dst_sort, CV_SORT_EVERY_ROW|CV_SORT_ASCENDING); //行列の各行を昇順にソートする
		//cout << "dst_sort=" << endl << dst_sort << endl;

		double med[10] = {};
		for (int h = 0; h < lnum; h++){ // lnum=10
			if (tnum % 2 == 1){ //データ数(列数)が奇数個の場合
				med[h] = dst_sort.at<double>(h, (tnum-1)/2);
			}
			else { //データ数(列数)が偶数個の場合
				med[h] = (dst_sort.at<double>(h, (tnum / 2) - 1) + dst_sort.at<double>(h, tnum / 2)) / 2.0;
			}
			cout << h << ".med= " << med[h] << endl;
		}

		//最小値を求める
		double min = 9999; //最小値を代入する変数
		int index = 0;
		for (int i = 0; i < lnum; i++){
			if (med[i] < min){
				min = med[i];
				index = i;			
			}
		}
		cout << "min= " << min << ",index= " << index << endl; //最小値とその場所

		// 近似直線までの距離がth以上の列先端セグメントとそれに続くセグメントを除去する
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
				//if (rect_width >= 10){ //横幅があまりにも小さいものは除外
					cv::rectangle(edge, cv::Point(lft, top), cv::Point(rgt, btm), cv::Scalar(0, 0, 200), 3, 4);
				//}
			}
		}

		/* 結果画像の表示 */
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
	// ウィンドウの破棄
	destroyAllWindows();
	return hr;
}