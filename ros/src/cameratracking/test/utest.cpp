#include <gtest/gtest.h>
#include <cameratracking.hpp>

cv::Size defaultSize()
{
    return cv::Size(640, 480);
}

cv::Mat defaultK()
{
    cv::Mat K = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    K.at<double>(0, 0) = 377.929565;//fx
    K.at<double>(1, 1) = 376.949158;//fy
    K.at<double>(0, 2) = 315.105469;//ppx
    K.at<double>(1, 2) = 250.341080;//ppy
    K.at<double>(2, 2) = 1;
    return K;
}

void example(cv::Mat& img, cv::Mat& depth, int d = 2000)
{
    img = cv::Mat(defaultSize(), CV_8UC3);
    depth = cv::Mat(defaultSize(), CV_16UC1);

    cv::Rect rect(img.size().width/4, img.size().height/4, img.size().width/2, img.size().height/2);
    cv::Mat roi = img(rect);
    cv::Mat roiD = depth(rect);

    for (int i=0; i<roi.size().height; i++) {
        for (int j=0; j<roi.size().width; j++) {
            roi.at<cv::Vec3b>(i, j) = cv::Vec3b(j%256, 0, i%256);
            roiD.at<unsigned short>(i, j) = d;
        }
    }
}

void show(const Pose3d& pose, std::string title = "")
{
    //ROS_INFO("%s trans: %f %f %f", title.c_str(), p.x(), p.y(), p.z());
    //ROS_INFO("%s rot: %f %f %f | %f", title.c_str(), q.x(), q.y(), q.z(), q.w());
    std::cout << pose.p.transpose() << std::endl;
    std::cout << pose.q.toRotationMatrix() << " " << std::endl;
}

struct F4 {
    template <typename T>
        bool operator()(const T* const x, T* residual) const {
            std::cout << "check : " << x[0]  << std::endl;
            residual[0] = (x[0]  - 100000) * (x[0] - 100000);
            return true;
        }
};


TEST(test, simple)
{
    double x =  0;
    ceres::Problem problem;

    ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<F4, ceres::CENTRAL, 1, 1>(new F4());
    ceres::LossFunction* loss_function = NULL;

    problem.AddResidualBlock(cost_function, loss_function, &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 3;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.eta = 1e-2;
    //options.max_solver_time_in_seconds = 100;
    //options.use_nonmonotonic_steps = false;
    //options.use_inner_iterations = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}
/*
TEST(test, track)
{
    // ピクセル座標系の再投影になっている
    cv::Mat K = defaultK();
    cv::Mat img0;
    cv::Mat depth;
    example(img0, depth);
    //Pose3d pose_correct(-1000,0,0, 0,1,0,M_PI/9);
    Pose3d pose_correct(-100,0,0, 0,0,1,0);
    show(pose_correct);
    cv::Mat img1 = reprojectImage(img0, depth, K, pose_correct);
    showImg("example", img0, false, -1);
    showImg("example", img1, false, -1);

    Pose3d pose0;
    Pose3d pose1_init;
    Pose3d pose1_est= track(img0, img1, depth, depth, K, K, pose0, pose1_init);

    show(pose1_est);
}
*/

// OK
/*
TEST(test, reprojection)
{
    // ピクセル座標系の再投影になっている
    cv::Mat K = defaultK();
    cv::Mat src;
    cv::Mat depth;
    example(src, depth);
    //Pose3d pose(1000,0,0, 0,0,1,0);
    //Pose3d pose(0, 1000,0, 0,0,1,0);
    //Pose3d pose(0, 0,500, 0,0,1,0);
    //Pose3d pose(0,0,0, 0,0,1,M_PI/6); // カメラが時計周りに30度回転する画像は反時計周りに回転する
    //Pose3d pose(0,0,0, 0,1,0,M_PI/9);
    Pose3d pose(-1000,0,0, 0,1,0,M_PI/9);
    cv::Mat dst = reprojectImage(src, depth, K, pose);

    //showImg("example", src, false, -1);
    //showImg("example", dst, false, -1);

    TrackingErrorTerm f(src, dst, depth, depth, K, K);

    double error = 0;
    f((double*)pose.p.data(), (double*)pose.q.coeffs().data(), &error);
    std::cout << error << std::endl;

    Pose3d pose1(0,0,0, 0,0,1,0);
    error = 0;
    f((double*)pose1.p.data(), (double*)pose1.q.coeffs().data(), &error);
    std::cout << error << std::endl;
}

TEST(test, projection)
{
    cv::Mat K = defaultK();

    //Eigen::Vector3d point(300, 200, 1000);
    Eigen::Vector3d point(0, 0, 1000);
    Eigen::Vector2d pxl;

    project(&pxl(0), &point(0), &K.at<double>(0,0));
    std::cout << pxl << std::endl;
    deproject(&point(0), &pxl(0), 1000.0, &K.at<double>(0,0));
    std::cout << point << std::endl;
}

TEST(test, transform)
{
    Pose3d world;
    Pose3d base(1,1,1, 0,0,1,M_PI/2);
    Pose3d target(2,2,2, 0,0,1,-M_PI/2);
    Pose3d rel = relativeTo(target, base);

    Eigen::Vector3d src(1, 1, 1);
    Eigen::Vector3d dst;
    transform(dst, rel.p, rel.q, src);
    std::cout << dst.transpose() << std::endl;
    show(rel);
    show(compose(base, rel));
    show(target);
}


TEST(test, subpxl)
{
    cv::Mat m(3, 3, CV_64FC1);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            m.at<double>(i, j) = i*3+j;
        }
    }

    bool ret;
    double v;
    ret = subpxl(v, m,  0.5, 0.5); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  1.5, 0.5); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,    1, 0.5); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  0.5, 1); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  1.5, 1); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,    1, 1); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  1.5, 1.5); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  0.4, 1.5); printf("%d : %f\n", ret, v);
    ret = subpxl(v, m,  2.5, 1.5); printf("%d : %f\n", ret, v);
}

TEST(OpenCVTutorial, check)
{
    cv::Mat m0(5, 6, CV_64FC3);
    std::cout << "  dims: " << m0.dims << ", depth(byte/channel):" << m0.elemSize1() << ", channels: " << m0.channels() << std::endl;
    std::cout << "  width: " << m0.size().width << "height:" << m0.size().height << std::endl;

    cv::Mat m(3, 3, CV_64FC3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            m.at<cv::Vec3d>(i, j) = cv::Vec3d(i, j, 3);
        }
    }

    double* p = (double*)m.data;
    for (int i=0; i<3*3*3; i++) {
        printf("%d : %f\n", i, p[i]);
    }
}
*/

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
