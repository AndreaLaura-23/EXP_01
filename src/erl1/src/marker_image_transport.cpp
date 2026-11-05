
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "image_transport/image_transport.hpp"

class MarkerImageTransport : public rclcpp::Node
{
public:
  MarkerImageTransport()
  : Node("marker_image_transport")
  {
    this->declare_parameter<std::string>(
      "input_topic", "/marker_image_raw");

    this->declare_parameter<std::string>(
      "output_topic", "/marker_image");

    const auto input_topic =
      this->get_parameter("input_topic").as_string();

    const auto output_topic =
      this->get_parameter("output_topic").as_string();

    publisher_ = image_transport::create_publisher(
      this, output_topic);

    subscription_ = this->create_subscription<
      sensor_msgs::msg::Image>(
        input_topic,
        rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg)
        {
          publisher_.publish(*msg);
        });

    RCLCPP_INFO(
      this->get_logger(),
      "Image transport: %s -> %s",
      input_topic.c_str(),
      output_topic.c_str());
  }

private:
  image_transport::Publisher publisher_;

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<MarkerImageTransport>());

  rclcpp::shutdown();
  return 0;
}
