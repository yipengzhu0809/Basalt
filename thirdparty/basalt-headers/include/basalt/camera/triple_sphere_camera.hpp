#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::sqrt;

/// @brief Double Sphere camera model
///
/// \image html ds.png
/// This model has N=6 parameters \f$ \mathbf{i} = \left[f_x, f_y, c_x, c_y,
/// \xi, \alpha \right]^T \f$ with \f$ \xi \in [-1,1], \alpha \in [0,1] \f$. See
/// \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class TripleSphereCamera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 7;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;
  using Mat3N = Eigen::Matrix<Scalar, 3, N>;

  /// @brief Default constructor with zero intrinsics
  TripleSphereCamera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, xi, alpha]
  explicit TripleSphereCamera(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  TripleSphereCamera<Scalar2> cast() const {
    return TripleSphereCamera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "ds"
  static std::string getName() { return "ts"; }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///    \pi(\mathbf{x}, \mathbf{i}) &=
  ///    \begin{bmatrix}
  ///    f_x{\frac{x}{\alpha d_2 + (1-\alpha) (\xi d_1 + z)}}
  ///    \\ f_y{\frac{y}{\alpha d_2 + (1-\alpha) (\xi d_1 + z)}}
  ///    \\ \end{bmatrix}
  ///    +
  ///    \begin{bmatrix}
  ///    c_x
  ///    \\ c_y
  ///    \\ \end{bmatrix},
  ///    \\ d_1 &= \sqrt{x^2 + y^2 + z^2},
  ///    \\ d_2 &= \sqrt{x^2 + y^2 + (\xi  d_1 + z)^2}.
  /// \f}
  /// A set of 3D points that results in valid projection is expressed as
  /// follows: \f{align}{
  ///    \Omega &= \{\mathbf{x} \in \mathbb{R}^3 ~|~ z > -w_2 d_1 \}
  ///    \\ w_2 &= \frac{w_1+\xi}{\sqrt{2w_1\xi + \xi^2 + 1}}
  ///    \\ w_1 &= \begin{cases} \frac{\alpha}{1-\alpha}, & \mbox{if } \alpha
  ///    \le 0.5 \\ \frac{1-\alpha}{\alpha} & \mbox{if } \alpha > 0.5
  ///    \end{cases}
  /// \f}
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
  template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d,
                      Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D d_proj_d_p3d = nullptr,
                      DerivedJparam d_proj_d_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
                                DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar& xi = param_[4];
    const Scalar& alpha = param_[5];
    const Scalar& lambda = param_[6];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;

    const Scalar r2 = xx + yy;

    const Scalar d1_2 = r2 + zz;
    const Scalar d1 = sqrt(d1_2);

    const Scalar w1 = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha
                                          : alpha / (Scalar(1) - alpha);
    const Scalar w2 =
        (w1 + xi + lambda) / sqrt(Scalar(1)+(xi+lambda)*(xi+lambda)+Scalar(2)*w1*(xi+lambda));

    const bool is_valid = (z > -w2 * d1);

    const Scalar k = xi * d1 + z;
    const Scalar kk = k * k;

    const Scalar d2_2 = r2 + kk;
    const Scalar d2 = sqrt(d2_2);

    const Scalar j = k + lambda * d2;
    const Scalar jj = j*j;

    const Scalar d3_2 = r2 + jj;
    const Scalar d3 = sqrt(d3_2);

    // const Scalar norm = alpha * d2 + (Scalar(1) - alpha) * k;

    // const Scalar mx = x / norm;
    // const Scalar my = y / norm;

    // proj[0] = fx * mx + cx; //TODO need to be remove later
    // proj[1] = fy * my + cy; //TODO need to be remove later

    const Scalar S = z + xi*d1 + lambda*d2 + (alpha/(Scalar(1)-alpha))*d3;

    proj[0] = (Scalar(1)/S)*(fx*x+cx*S);
    proj[1] = (Scalar(1)/S)*(fy*y+cy*S);

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);

      // const Scalar norm2 = norm * norm;
      // const Scalar xy = x * y;
      // const Scalar tt2 = xi * z / d1 + Scalar(1);

      // const Scalar d_norm_d_r2 = (xi * (Scalar(1) - alpha) / d1 +
      //                             alpha * (xi * k / d1 + Scalar(1)) / d2) /
      //                            norm2;

      // const Scalar tmp2 =
      //     ((Scalar(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;
      
      const Scalar d1dx = x / d1;
      const Scalar d2dx = (x+(xi*d1+z)*xi*d1dx)/ d2;
      const Scalar d3dx = (x+j*(lambda*d2dx+xi*d1dx)) / d3;
      const Scalar dSdx = xi*d1dx+lambda*d2dx+(alpha/(Scalar(1)-alpha))*d3dx;
      const Scalar dudx = (fx*S-fx*x*dSdx)/(S*S);

      const Scalar d1dy = y / d1;
      const Scalar d2dy = (y + (xi*d1+z)*xi*d1dy)/d2;
      const Scalar d3dy = (y+(lambda*d2+xi*d1+z)*(lambda*d2dy+xi*d1dy))/d3;
      const Scalar dSdy = xi*d1dy+lambda*d2dy+(alpha/(Scalar(1)-alpha))*d3dy;
      const Scalar dudy = (Scalar(-1)*dSdy*fx*x)/(S*S);

      const Scalar d1dz = z / d1;
      const Scalar d2dz = (xi*d1+z)*(xi*d1dz+Scalar(1))/d2;
      const Scalar d3dz = (lambda*d2dz+xi*d1dz+Scalar(1))*(lambda*d2+xi*d1+z)/d3;
      const Scalar dSdz = Scalar(1)+xi*d1dz+lambda*d2dz+(alpha/(Scalar(1)-alpha))*d3dz;
      const Scalar dudz = (Scalar(-1)*dSdz*fx*x)/(S*S);

      const Scalar dvdx = Scalar(-1)*dSdx*fy*y/(S*S);
      const Scalar dvdy = (fy*S-dSdy*fy*y)/(S*S);
      const Scalar dvdz = Scalar(-1)*dSdz*fy*y/(S*S);



      d_proj_d_p3d->setZero();
      // (*d_proj_d_p3d)(0, 0) = fx * (Scalar(1) / norm - xx * d_norm_d_r2);
      // (*d_proj_d_p3d)(1, 0) = -fy * xy * d_norm_d_r2;

      // (*d_proj_d_p3d)(0, 1) = -fx * xy * d_norm_d_r2;
      // (*d_proj_d_p3d)(1, 1) = fy * (Scalar(1) / norm - yy * d_norm_d_r2);

      // (*d_proj_d_p3d)(0, 2) = -fx * x * tmp2;
      // (*d_proj_d_p3d)(1, 2) = -fy * y * tmp2;

      (*d_proj_d_p3d)(0, 0) = dudx;
      (*d_proj_d_p3d)(0, 1) = dudy;
      (*d_proj_d_p3d)(0, 2) = dudz;
      (*d_proj_d_p3d)(1, 0) = dvdx;
      (*d_proj_d_p3d)(1, 1) = dvdy;
      (*d_proj_d_p3d)(1, 2) = dvdz;

    } else {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);

      // const Scalar norm2 = norm * norm;

      (*d_proj_d_param).setZero();
      // (*d_proj_d_param)(0, 0) = mx;
      // (*d_proj_d_param)(0, 2) = Scalar(1);
      // (*d_proj_d_param)(1, 1) = my;
      // (*d_proj_d_param)(1, 3) = Scalar(1);

      // const Scalar tmp4 = (alpha - Scalar(1) - alpha * k / d2) * d1 / norm2;
      // const Scalar tmp5 = (k - d2) / norm2;

      // (*d_proj_d_param)(0, 4) = fx * x * tmp4;
      // (*d_proj_d_param)(1, 4) = fy * y * tmp4;

      // (*d_proj_d_param)(0, 5) = fx * x * tmp5;
      // (*d_proj_d_param)(1, 5) = fy * y * tmp5;

      const Scalar d2dxi = k*d1/d2;
      const Scalar d3dla = j*d2/d3;
      (*d_proj_d_param)(0, 0) = x/S;
      (*d_proj_d_param)(0, 1) = Scalar(0);
      (*d_proj_d_param)(0, 2) = Scalar(1);
      (*d_proj_d_param)(0, 3) = Scalar(0);
      (*d_proj_d_param)(0, 4) = (Scalar(-1)*fx*x/(S*S))*(d1+lambda*d2dxi+(alpha/(Scalar(1)-alpha))*j*(d1+lambda*d2dxi)/d3);
      (*d_proj_d_param)(0, 5) = (Scalar(-1)*fx*x/(S*S))*d3*(Scalar(1)/((Scalar(1)-alpha)*(Scalar(1)-alpha)));
      (*d_proj_d_param)(0, 6) = (Scalar(-1)*fx*x/(S*S))*(d2+(alpha/(Scalar(1)-alpha))*d3dla);

      (*d_proj_d_param)(1, 0) = Scalar(0);
      (*d_proj_d_param)(1, 1) = y/S;
      (*d_proj_d_param)(1, 2) = Scalar(0);
      (*d_proj_d_param)(1, 3) = Scalar(1);
      (*d_proj_d_param)(1, 4) = (Scalar(-1)*fy*y/(S*S))*(d1+lambda*d2dxi+(alpha/(Scalar(1)-alpha))*j*(d1+lambda*d2dxi)/d3);
      (*d_proj_d_param)(1, 5) = (Scalar(-1)*fy*y/(S*S))*d3*(Scalar(1)/((Scalar(1)-alpha)*(Scalar(1)-alpha)));
      (*d_proj_d_param)(1, 6) = (Scalar(-1)*fy*y/(S*S))*(d2+(alpha/(Scalar(1)-alpha))*d3dla);
    } else {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Unproject the point and optionally compute Jacobians
  ///
  /// The unprojection function is computed as follows: \f{align}{
  ///    \pi^{-1}(\mathbf{u}, \mathbf{i}) &=
  ///    \frac{m_z \xi + \sqrt{m_z^2 + (1 - \xi^2) r^2}}{m_z^2 + r^2}
  ///    \begin{bmatrix}
  ///    m_x \\ m_y \\m_z
  ///    \\ \end{bmatrix}-\begin{bmatrix}
  ///    0 \\ 0 \\ \xi
  ///    \\ \end{bmatrix},
  ///    \\ m_x &= \frac{u - c_x}{f_x},
  ///    \\ m_y &= \frac{v - c_y}{f_y},
  ///    \\ r^2 &= m_x^2 + m_y^2,
  ///    \\ m_z &= \frac{1 - \alpha^2  r^2}{\alpha  \sqrt{1 - (2 \alpha - 1)
  ///    r^2}
  ///    + 1 - \alpha},
  /// \f}
  ///
  /// The valid range of unprojections is \f{align}{
  ///    \Theta &= \begin{cases}
  ///    \mathbb{R}^2 & \mbox{if } \alpha \le 0.5
  ///    \\ \{ \mathbf{u} \in \mathbb{R}^2 ~|~ r^2 \le \frac{1}{2\alpha-1} \}  &
  ///    \mbox{if} \alpha > 0.5 \end{cases}
  /// \f}
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj if not nullptr computed Jacobian of unprojection
  /// with respect to proj
  /// @param[out] d_p3d_d_param point if not nullptr computed Jacobian of
  /// unprojection with respect to intrinsic parameters
  /// @return if unprojection is valid
  template <class DerivedPoint2D, class DerivedPoint3D,
            class DerivedJ2D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool unproject(const Eigen::MatrixBase<DerivedPoint2D>& proj,
                        Eigen::MatrixBase<DerivedPoint3D>& p3d,
                        DerivedJ2D d_p3d_d_proj = nullptr,
                        DerivedJparam d_p3d_d_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D,
                                  DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint2D>::Type proj_eval(proj);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar& xi = param_[4];
    const Scalar& alpha = param_[5];
    const Scalar& lambda = param_[6];

    const Scalar mx = (proj_eval[0] - cx) / fx;
    const Scalar my = (proj_eval[1] - cy) / fy;

    const Scalar u = proj_eval[0];
    const Scalar v = proj_eval[1];

    const Scalar S = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha
                                          : alpha / (Scalar(1) - alpha);
    const Scalar& gamma = (S+sqrt(Scalar(1)+(1-S*S)*(mx*mx+my*my)))/(mx*mx+my*my+Scalar(1));
    const Scalar& eta = lambda*(gamma-S)+sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+1);

    

    const Scalar r2 = mx * mx + my * my;

    //const Scalar tmp3 = (sqrt(Scalar(1)+(Scalar(1)-S*S)*r2) / (r2+Scalar(1)))-((S*r2)/(r2+Scalar(1)));
    // const Scalar tmp4 = (eta*(gamma-S)-lambda)*(eta*(gamma-S)-lambda);

/*    !static_cast<bool>(alpha > Scalar(0.5) &&
                           (r2 >= Scalar(1) / (Scalar(2) * alpha - Scalar(1))));*/    

    const Scalar xi2_2 = alpha * alpha;
    const Scalar xi1_2 = xi * xi;

    const Scalar sqrt2 = sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) * r2);

    const Scalar norm2 = alpha * sqrt2 + Scalar(1) - alpha;

    const Scalar mz1 = (Scalar(1) - xi2_2 * r2) / norm2;
    const Scalar mz = eta*(gamma-S)-lambda+mz1-mz1;
    const Scalar mz2 = mz * mz;

    const Scalar norm1 = mz2 + r2;
    const Scalar sqrt1 = sqrt(mz2 + (Scalar(1) - xi1_2) * r2);
    const Scalar k = (mz * xi + sqrt1) / norm1;

    const Scalar mu = xi*mz+sqrt(xi*xi*mz*mz-xi*xi+Scalar(1));

    // const bool is_valid = static_cast<bool>((r2 >= Scalar(1)/(S*S-Scalar(1))) && (lambda*lambda*tmp3*tmp3 >= lambda * lambda -Scalar(1)) && (xi*xi*tmp4-xi*xi+Scalar(1)));
    const bool is_valid = static_cast<bool>((Scalar(1)+(Scalar(1)-S*S)*r2) >= Scalar(0) && (lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1)) >= Scalar(0) && (xi*xi*mz*mz-xi*xi+Scalar(1))>=Scalar(0));

    p3d.setZero();
    p3d[0] = k * mx;
    p3d[1] = k * my;
    p3d[2] = k * mz - xi;

    p3d[0] = mu * eta * gamma * mx;
    p3d[1] = mu * eta * gamma * my;
    p3d[2] = mu * mz - xi + u - u + v - v;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> ||
                  !std::is_same_v<DerivedJparam, std::nullptr_t>) {
      const Scalar norm2_2 = norm2 * norm2;
      const Scalar norm1_2 = norm1 * norm1;

      const Scalar d_mz_d_r2 = (Scalar(0.5) * alpha - xi2_2) *
                                   (r2 * xi2_2 - Scalar(1)) /
                                   (sqrt2 * norm2_2) -
                               xi2_2 / norm2;

      const Scalar d_mz_d_mx = 2 * mx * d_mz_d_r2;
      const Scalar d_mz_d_my = 2 * my * d_mz_d_r2;

      const Scalar d_k_d_mz =
          (norm1 * (xi * sqrt1 + mz) - 2 * mz * (mz * xi + sqrt1) * sqrt1) /
          (norm1_2 * sqrt1);

      const Scalar d_k_d_r2 =
          (xi * d_mz_d_r2 +
           Scalar(0.5) / sqrt1 *
               (Scalar(2) * mz * d_mz_d_r2 + Scalar(1) - xi1_2)) /
              norm1 -
          (mz * xi + sqrt1) * (Scalar(2) * mz * d_mz_d_r2 + Scalar(1)) /
              norm1_2;

      const Scalar d_k_d_mx = d_k_d_r2 * 2 * mx;
      const Scalar d_k_d_my = d_k_d_r2 * 2 * my;

      constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
      Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;

      c0.setZero();
      c0[0] = (mx * d_k_d_mx + k);
      c0[1] = my * d_k_d_mx;
      c0[2] = (mz * d_k_d_mx + k * d_mz_d_mx);
      c0 /= fx;

      c1.setZero();
      c1[0] = mx * d_k_d_my;
      c1[1] = (my * d_k_d_my + k);
      c1[2] = (mz * d_k_d_my + k * d_mz_d_my);
      c1 /= fy;



      const Scalar dmxdu = Scalar(1)/fx;
      const Scalar dmxdv = Scalar(0);
      const Scalar dmydu = Scalar(0);
      const Scalar dmydv = Scalar(1)/fy;
      const Scalar tmp1 = mx*mx+my*my+Scalar(1);
      const Scalar tmp2 = sqrt(Scalar(1)+(Scalar(1)-S*S)*(mx*mx+my*my));
      const Scalar dgadu = ((((1-S*S)*mx/fx)*tmp1/tmp2)-((tmp2+S)*Scalar(2)*mx/fx))/(tmp1*tmp1);
      const Scalar detdu = lambda*dgadu+ (lambda*lambda*(gamma-S)*dgadu)/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
      const Scalar dmzdu = detdu*(gamma-S)+dgadu*eta;
      const Scalar dmudu = xi*dmzdu+(xi*xi*mz*dmzdu)/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));
      const Scalar dxdu = dmudu*eta*gamma*mx+mu*(detdu*gamma*mx+eta*(dgadu*mx+dmxdu*gamma));

      const Scalar dgadv = ((((1-S*S)*my/fy)*tmp1/tmp2)-((tmp2+S)*Scalar(2)*my/fy))/(tmp1*tmp1);
      const Scalar detdv = lambda*dgadv+ (lambda*lambda*(gamma-S)*dgadv)/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
      const Scalar dmzdv = detdv*(gamma-S)+dgadv*eta;
      const Scalar dmudv = xi*dmzdv+(xi*xi*mz*dmzdv)/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));
      const Scalar dxdv = dmudv*eta*gamma*mx+mu*(detdv*gamma*mx+eta*(dgadv*mx+dmxdv*gamma));

      const Scalar dydu = dmudu*eta*gamma*my+mu*(detdu*gamma*my+eta*(dgadu*my+dmydu*gamma));
      const Scalar dydv = dmudv*eta*gamma*my+mu*(detdv*gamma*my+eta*(dgadv*my+dmydv*gamma));

      const Scalar dzdu = dmudu*mz+dmzdu*mu;
      const Scalar dzdv = dmudv*mz+dmzdv*mu;

      if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
        BASALT_ASSERT(d_p3d_d_proj);
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
        (d_p3d_d_proj)(0,0) = dxdu;
        (d_p3d_d_proj)(0,1) = dxdv;
        (d_p3d_d_proj)(1,0) = dydu;
        (d_p3d_d_proj)(1,1) = dydv;
        (d_p3d_d_proj)(2,0) = dzdu;
        (d_p3d_d_proj)(2,1) = dzdv;
      } else {
        UNUSED(d_p3d_d_proj);
      }

      if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
        BASALT_ASSERT(d_p3d_d_param);
        const Scalar d_k_d_xi1 = (mz * sqrt1 - xi * r2) / (sqrt1 * norm1);

        const Scalar d_mz_d_xi2 =
            ((Scalar(1) - r2 * xi2_2) *
                 (r2 * alpha / sqrt2 - sqrt2 + Scalar(1)) / norm2 -
             Scalar(2) * r2 * alpha) /
            norm2;

        const Scalar d_k_d_xi2 = d_k_d_mz * d_mz_d_xi2;

        const Scalar dmxdfx = (cx - u)/(fx*fx);
        const Scalar dmxdcx = Scalar(-1)/fx;
        const Scalar dmydfy = (cy - v)/(fy*fy);
        const Scalar dmydcy = Scalar(-1)/fy;

        const Scalar dgadfx = (((tmp1*(1-S*S)*mx*dmxdfx)/tmp2)-(S+tmp2)*Scalar(2)*mx*dmxdfx)/(tmp1*tmp1);
        const Scalar detdfx = lambda*dgadfx + lambda*lambda*(gamma-S)*dgadfx/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
        const Scalar dmzdfx = detdfx*(gamma-S)+dgadfx*eta;
        const Scalar dmudfx = xi*dmzdfx+ xi*xi*mz*dmzdfx/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));

        const Scalar dgadfy = (((tmp1*(1-S*S)*my*dmydfy)/tmp2)-(S+tmp2)*Scalar(2)*my*dmydfy)/(tmp1*tmp1);
        const Scalar detdfy = lambda*dgadfy + lambda*lambda*(gamma-S)*dgadfy/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
        const Scalar dmzdfy = detdfy*(gamma-S)+dgadfy*eta;
        const Scalar dmudfy = xi*dmzdfy+ xi*xi*mz*dmzdfy/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));

        const Scalar dgadcx = (((tmp1*(1-S*S)*mx*dmxdcx)/tmp2)-(S+tmp2)*Scalar(2)*mx*dmxdcx)/(tmp1*tmp1);
        const Scalar detdcx = lambda*dgadcx + lambda*lambda*(gamma-S)*dgadcx/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
        const Scalar dmzdcx = detdcx*(gamma-S)+dgadcx*eta;
        const Scalar dmudcx = xi*dmzdcx+ xi*xi*mz*dmzdcx/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));

        const Scalar dgadcy = (((tmp1*(1-S*S)*my*dmydcy)/tmp2)-(S+tmp2)*Scalar(2)*my*dmydcy)/(tmp1*tmp1);
        const Scalar detdcy = lambda*dgadcy + lambda*lambda*(gamma-S)*dgadcy/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
        const Scalar dmzdcy = detdcy*(gamma-S)+dgadcy*eta;
        const Scalar dmudcy = xi*dmzdcy+ xi*xi*mz*dmzdcy/(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));

        const Scalar dmudxi = mz+ (mz*mz*xi-xi)/sqrt(xi*xi*mz*mz-xi*xi+Scalar(1));

        const Scalar dSdal = alpha > Scalar(0.5) ? Scalar(-1) / (alpha*alpha)
                                          : Scalar(1) / (Scalar(1) - alpha)*(Scalar(1) - alpha);
        const Scalar dgadal = (1/tmp1)*(dSdal-(mx*mx+my*my)*S*dSdal/(sqrt(tmp2)));
        const Scalar detdal = lambda*(dgadal-dSdal)+ lambda*lambda*(gamma-S)*(dgadal-dSdal)/sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1));
        const Scalar dmzdal = detdal*(gamma-S)+eta*(dgadal-dSdal);
        const Scalar dmudal = xi*dmzdal + xi*xi*mz*dmzdal /(sqrt(Scalar(1)+xi*xi*mz*mz-xi*xi));

        const Scalar detdla = (gamma-S) + ((gamma-S)*(gamma-S)*lambda - lambda)/(sqrt(lambda*lambda*(gamma-S)*(gamma-S)-lambda*lambda+Scalar(1)));
        const Scalar dmzdla = (gamma-S)*detdla-Scalar(1);
        const Scalar dmudla = xi*dmzdla+ xi*xi*mz*dmzdla / (sqrt(xi*xi*mz*mz-xi*xi+Scalar(1)));

        const Scalar dxdfx = dmudfx*eta*gamma*mx+mu*(detdfx*gamma*mx+eta*(dgadfx*mx+dmxdfx*gamma));
        const Scalar dxdfy = dmudfy*eta*gamma*mx+mu*(detdfy*gamma*mx+eta*(dgadfy*mx));
        const Scalar dxdcx = dmudcx*eta*gamma*mx+mu*(detdcx*gamma*mx+eta*(dgadcx*mx+dmxdcx*gamma));
        const Scalar dxdcy = dmudcy*eta*gamma*mx+mu*(detdcy*gamma*mx+eta*(dgadcy*mx));
        const Scalar dxdxi = eta*gamma*mx*dmudxi;
        const Scalar dxdal = dmudal*eta*gamma*mx+mu*(detdal*gamma*mx+eta*(dgadal*mx));
        const Scalar dxdla = gamma*(dmudla*eta*mx+mu*(detdla*mx));

        const Scalar dydfx = dmudfx*eta*gamma*my+mu*(detdfx*gamma*my+eta*(dgadfx*my));
        const Scalar dydfy = dmudfy*eta*gamma*my+mu*(detdfy*gamma*my+eta*(dgadfy*my+dmydfy*gamma));
        const Scalar dydcx = dmudcx*eta*gamma*my+mu*(detdcx*gamma*my+eta*(dgadcx*my));
        const Scalar dydcy = dmudcy*eta*gamma*my+mu*(detdcy*gamma*my+eta*(dgadcy*my+dmydcy*gamma));
        const Scalar dydxi = eta*gamma*my*dmudxi;
        const Scalar dydal = dmudal*eta*gamma*my+mu*(detdal*gamma*my+eta*(dgadal*my));
        const Scalar dydla = gamma*(dmudla*eta*my+mu*(detdla*my));

        const Scalar dzdfx = dmudfx*mz+dmzdfx*mu;
        const Scalar dzdfy = dmudfy*mz+dmzdfy*mu;
        const Scalar dzdcx = dmudcx*mz+dmzdcx*mu;
        const Scalar dzdcy = dmudcy*mz+dmzdcy*mu;
        const Scalar dzdxi = dmudxi*mz-Scalar(1);
        const Scalar dzdal = dmudal*mz+dmzdal*mu;
        const Scalar dzdla = dmudla*mz+dmzdla*mu;


        d_p3d_d_param->setZero();
        (*d_p3d_d_param).col(0) = -c0 * mx;
        (*d_p3d_d_param).col(1) = -c1 * my;

        (*d_p3d_d_param).col(2) = -c0;
        (*d_p3d_d_param).col(3) = -c1;

        (*d_p3d_d_param)(0, 4) = mx * d_k_d_xi1;
        (*d_p3d_d_param)(1, 4) = my * d_k_d_xi1;
        (*d_p3d_d_param)(2, 4) = mz * d_k_d_xi1 - 1;

        (*d_p3d_d_param)(0, 5) = mx * d_k_d_xi2;
        (*d_p3d_d_param)(1, 5) = my * d_k_d_xi2;
        (*d_p3d_d_param)(2, 5) = mz * d_k_d_xi2 + k * d_mz_d_xi2;
        
        (*d_p3d_d_param)(0,0) = dxdfx;
        (*d_p3d_d_param)(0,1) = dxdfy;
        (*d_p3d_d_param)(0,2) = dxdcx;
        (*d_p3d_d_param)(0,3) = dxdcy;
        (*d_p3d_d_param)(0,4) = dxdxi;
        (*d_p3d_d_param)(0,5) = dxdal;
        (*d_p3d_d_param)(0,6) = dxdla;
        (*d_p3d_d_param)(1,0) = dydfx;
        (*d_p3d_d_param)(1,1) = dydfy;
        (*d_p3d_d_param)(1,2) = dydcx;
        (*d_p3d_d_param)(1,3) = dydcy;
        (*d_p3d_d_param)(1,4) = dydxi;
        (*d_p3d_d_param)(1,5) = dydal;
        (*d_p3d_d_param)(1,6) = dydla;
        (*d_p3d_d_param)(2,0) = dzdfx;
        (*d_p3d_d_param)(2,1) = dzdfy;
        (*d_p3d_d_param)(2,2) = dzdcx;
        (*d_p3d_d_param)(2,3) = dzdcy;
        (*d_p3d_d_param)(2,4) = dzdxi;
        (*d_p3d_d_param)(2,5) = dzdal;
        (*d_p3d_d_param)(2,6) = dzdla;

      } else {
        UNUSED(d_p3d_d_param);
        UNUSED(d_k_d_mz);
      }
    } else {
      UNUSED(d_p3d_d_proj);
      UNUSED(d_p3d_d_param);
    }

    return is_valid;
  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, 0, 0.5
  /// \right]^T \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    param_[0] = init[0];
    param_[1] = init[1];
    param_[2] = init[2];
    param_[3] = init[3];
    param_[4] = 0;
    param_[5] = 0;
    param_[6] = 0;
  }

  /// @brief Increment intrinsic parameters by inc and clamp the values to the
  /// valid range
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) {
    param_ += inc;
    param_[4] = std::clamp(param_[4], Scalar(-1), Scalar(1));
    param_[5] = std::clamp(param_[5], Scalar(0), Scalar(1));
  }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, \xi, \alpha
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<TripleSphereCamera> getTestProjections() {
    Eigen::aligned_vector<TripleSphereCamera> res;

    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0.5 * 1.48785;
    res.emplace_back(vec1);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;
};

}  // namespace basalt
