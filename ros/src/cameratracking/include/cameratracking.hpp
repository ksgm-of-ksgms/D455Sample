#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


template <typename T> bool subpxl(T& v, const cv::Mat mat,  double y, double x)
{
    int x1 = round(x);
    int y1 = round(y);
    int x0 = x1 - 1;
    int y0 = y1 - 1;

    if (!(y0 >= 0 && x0 >= 0 && y1 < mat.size().height && x1 < mat.size().width)) {
        return false;
    }

    T pxl00 = mat.at<T>(y0, x0);
    T pxl01 = mat.at<T>(y0, x1);
    T pxl10 = mat.at<T>(y1, x0);
    T pxl11 = mat.at<T>(y1, x1);

    double kx = x + 0.5 - round(x);
    double ky = y + 0.5 - round(y);

    //printf("%f %f %f %f %f %f %f\n", kx, ky, pxl00, pxl01, pxl10, pxl11);
    v = (1-kx) * (1-ky) * pxl00 + (1-kx) * ky * pxl10 + kx * (1-ky) * pxl01 + kx * ky * pxl11;

    return true;
}

template <typename T> struct Pose3 {
    Eigen::Matrix<T, 3, 1> p;
    Eigen::Quaternion<T> q;
    static std::string name() { return "VERTEX_SE3:QUAT"; }
    Pose3() : q(T(1), T(0), T(0), T(0)) { }
    Pose3(T x, T y, T z, T ax, T ay, T az, T atheta)
    {
        q = Eigen::AngleAxis<T>(atheta, Eigen::Matrix<T, 3, 1>(ax, ay, az));
        p = Eigen::Matrix<T, 3, 1>(x,y,z);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


typedef Pose3<double> Pose3d;

static inline Pose3d relativeTo(const Pose3d& target, const Pose3d& base)
{
    Pose3d ret;
    auto bqinv = base.q.conjugate();
    ret.q = bqinv * target.q;
    ret.p = bqinv * (target.p - base.p);
    return ret;
}

static inline Pose3d compose(const Pose3d& base, const Pose3d& relative)
{
    Pose3d ret;
    ret.q = base.q * relative.q;
    ret.p = base.q * relative.p + base.p;
    return ret;
}

static inline void showImg(std::string label, const cv::Mat &mat ,bool depth = false, int waitMs = 1)
{
    cv::namedWindow(label.c_str());
    if (depth) {
        cv::Mat mat2;
        mat.convertTo(mat2, CV_8U, 1/256.0);
        cv::applyColorMap(mat2, mat2, cv::COLORMAP_JET);
        cv::imshow(label.c_str(), mat2);
    } else {
        cv::imshow(label.c_str(), mat);
    }
    cv::waitKey(waitMs);
}


static inline bool inRange(const cv::Size s, int x, int y)
{
    if (x < 0 || y < 0 || x >= s.width || y >= s.height) {
        return false;
    }
    return true;
}

template <typename T> void transform(
        Eigen::Matrix<T, 3, 1> &dst,
        const Eigen::Matrix<T, 3, 1> &p,
        const Eigen::Quaternion<T> &q,
        const Eigen::Matrix<T, 3, 1> &src)
{
    dst = q.conjugate() * (src - p);
}


// depth [mm]
template <typename T> void deproject(T* point, const T* pxl, const T depth, const T* K)
{
    point[0] = depth * (pxl[0] - K[2]) / K[0];
    point[1] = depth * (pxl[1] - K[5]) / K[4];
    point[2] = depth;
}


template <typename T> void project(T* pxl, const T* point, const T* K)
{
     pxl[0] = point[0] / point[2] * K[0] + K[2];
     pxl[1] = point[1] / point[2] * K[4] + K[5];
}

// for test
static inline cv::Mat reprojectImage(const cv::Mat& src, const cv::Mat& depth, const cv::Mat& K, const Pose3d pose, const unsigned short minDepth = 400)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            cv::Vec3b c = src.at<cv::Vec3b>(i,j);
            double d = (double) depth.at<unsigned short>(i, j);
            if (d <= minDepth) {
                continue;
            }

            double pxl[2];
            pxl[0] = double(j);
            pxl[1] = double(i);

            double point[3];
            deproject(point, pxl, d, &K.at<double>(0,0));
            const Eigen::Matrix<double, 3, 1> p00 = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(point);
            Eigen::Matrix<double, 3, 1> p01;
            transform(p01, pose.p, pose.q, p00);
            project(pxl, &p01.coeff(0, 0), &K.at<double>(0,0));

            if (inRange(dst.size(), round(pxl[0]), round(pxl[1]))) {
                //std::cout << j << " " << i << " " << pxl[0] << " " << pxl[1] << std::endl;
                dst.at<cv::Vec3b>(round(pxl[1]), round(pxl[0])) = c; // NN
            }
        }
    }
    return dst;
}


class TrackingErrorTerm {
    public:
        TrackingErrorTerm(
                const cv::Mat &rgb0,
                const cv::Mat &rgb1,
                const cv::Mat &depth0,
                const cv::Mat &depth1, // dummy
                const cv::Mat &rgbCameraMat0,
                const cv::Mat &rgbCameraMat1)
                : rgb0(rgb0), rgb1(rgb1), depth0(depth0), depth1(depth1), rgbCameraMat0(rgbCameraMat0), rgbCameraMat1(rgbCameraMat1)
        {
        }

        bool operator()(const double* const p_ptr, const double* const q_ptr, double* residuals_ptr) const {

            const Eigen::Matrix<double, 3, 1> p = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(p_ptr);
            const Eigen::Quaternion<double> q = Eigen::Map<const Eigen::Quaternion<double>>(q_ptr);

            double K0[9];
            double K1[9];
            for (int i=0; i<3; i++) {
                for (int j=0; j<3; j++) {
                    K0[3*i+j] = double(rgbCameraMat0.at<double>(i,j));
                    K1[3*i+j] = double(rgbCameraMat1.at<double>(i,j));
                }
            }

            double total_error = 0;
            for (int i = 0; i < rgb0.rows; i++) {
                for (int j = 0; j < rgb0.cols; j++) {
                    // 0 -> 1
                    double d0 = double(depth0.at<unsigned short>(i,j));
                    if (d0 < 400) {
                        //std::cout << i << " " << j << " : " << d0 << " skip by depth" << std::endl;
                        continue;
                    }
                    cv::Vec3b color0 = rgb0.at<cv::Vec3b>(i,j);

                    double pxl[2];
                    pxl[0] = double(j);
                    pxl[1] = double(i);

                    double point[3];
                    deproject(point, pxl, d0, K0);
                    const Eigen::Matrix<double, 3, 1> p00 = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(point);
                    Eigen::Matrix<double, 3, 1> p01;
                    transform(p01, p, q, p00);
                    project(pxl, &p01.coeff(0, 0), K1);

                    cv::Vec3b color1;
                    bool ret = subpxl(color1, rgb1, pxl[1], pxl[0]);
                    if (!ret) {
                        //std::cout << i << " " << j << " : skip out of image" << std::endl;
                        continue;
                    }
                    double pxl_error = (abs(color1[0] - color0[0]) + abs(color1[1] - color0[1]) + abs(color1[2] - color0[2])) / 256.0;
                    total_error += pxl_error;
                }
            }
            residuals_ptr[0] = total_error;

            return true;
        }

        static ceres::CostFunction* Create(
                const cv::Mat &rgb0,
                const cv::Mat &rgb1,
                const cv::Mat &depth0,
                const cv::Mat &depth1,
                const cv::Mat &rgbCameraMat0,
                const cv::Mat &rgbCameraMat1)
        {
            //return new ceres::AutoDiffCostFunction<TrackingErrorTerm, 1, 3, 4>(
            return new ceres::NumericDiffCostFunction<TrackingErrorTerm, ceres::CENTRAL, 1, 3, 4>(
                    new TrackingErrorTerm(rgb0, rgb1, depth0, depth1, rgbCameraMat0, rgbCameraMat1));
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        const cv::Mat &rgb0; // undistorted
        const cv::Mat &rgb1; // undistorted
        const cv::Mat &depth0; // undistorted
        const cv::Mat &depth1; // undistorted
        const cv::Mat &rgbCameraMat0;
        const cv::Mat &rgbCameraMat1;
};

//前フレームからみた次フレームの位置姿勢を求める
static inline Pose3d track(const cv::Mat &rgb0, const cv::Mat &rgb1, const cv::Mat &depth0, const cv::Mat &depth1, const cv::Mat &rgbCameraMat0, const cv::Mat &rgbCameraMat1, const Pose3d &pose0, const Pose3d &pose1_init)
{
    ceres::Problem problem;

    //Pose3d pose = relativeTo(pose1_init, pose0);
    Pose3d pose = pose1_init;

    auto localparameterization = new ceres::EigenQuaternionParameterization();

    ceres::CostFunction* cost_function =  TrackingErrorTerm::Create(rgb0, rgb1, depth0, depth1, rgbCameraMat0, rgbCameraMat1);

    ceres::LossFunction* loss_function = NULL;
    //ceres::LossFunction* loss_function = new HuberLoss(1.0);

    problem.AddResidualBlock(cost_function,
            loss_function,
            pose.p.data(),
            pose.q.coeffs().data());

    problem.SetParameterization(pose.q.coeffs().data(), localparameterization);

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

    //return compose(pose0, pose);
    return pose;
}
