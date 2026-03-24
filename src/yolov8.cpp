//
// Created by  triple-Mu     on 24-01-2023.
// Modified by Q-engineering on 06-03-2024
// Modified by inaciose      on 24-03-2026
//

// TensorRT 10 compatible version (JetPack 6.2.2)

#include "yolov8.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

//----------------------------------------------------------------------------------------
//using namespace det;
//----------------------------------------------------------------------------------------
const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    initLibNvInferPlugins(&this->gLogger, "");

    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream.data(), size);
    assert(this->engine);

    this->context = this->engine->createExecutionContext();
    assert(this->context);

    cudaStreamCreate(&this->stream);

    // NEW API
    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;

        const char* name = this->engine->getIOTensorName(i);
        binding.name = name;

        auto dtype = this->engine->getTensorDataType(name);
        binding.dsize = type_to_size(dtype);

        auto mode = this->engine->getTensorIOMode(name);

        nvinfer1::Dims dims = this->engine->getTensorShape(name);

        binding.dims = dims;
        binding.size = get_size_by_dims(dims);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs++;
            this->input_bindings.push_back(binding);
        } else {
            this->num_outputs++;
            this->output_bindings.push_back(binding);
        }
    }
}

YOLOv8::~YOLOv8()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;

    cudaStreamDestroy(this->stream);

    for (auto& ptr : this->device_ptrs) {
        cudaFree(ptr);
    }

    for (auto& ptr : this->host_ptrs) {
        cudaFreeHost(ptr);
    }
}

void YOLOv8::MakePipe(bool warmup)
{
    for (auto& binding : this->input_bindings) {
        void* d_ptr;
        cudaMalloc(&d_ptr, binding.size * binding.dsize);
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& binding : this->output_bindings) {
        void *d_ptr, *h_ptr;
        size_t size = binding.size * binding.dsize;

        cudaMalloc(&d_ptr, size);
        cudaHostAlloc(&h_ptr, size, 0);

        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // IMPORTANT: bind tensors
    int idx = 0;
    for (auto& b : this->input_bindings) {
        this->context->setTensorAddress(b.name.c_str(), this->device_ptrs[idx++]);
    }
    for (auto& b : this->output_bindings) {
        this->context->setTensorAddress(b.name.c_str(), this->device_ptrs[idx++]);
    }

    if (warmup) {
        for (int i = 0; i < 5; i++) {
            this->Infer();
        }
    }
}

void YOLOv8::CopyFromMat(const cv::Mat& image)
{
    auto& in = this->input_bindings[0];

    int width  = in.dims.d[3];
    int height = in.dims.d[2];

    cv::Mat nchw;
    cv::Size size(width, height);

    this->Letterbox(image, nchw, size);

    nvinfer1::Dims dims{4, {1, 3, height, width}};
    this->context->setInputShape(in.name.c_str(), dims);

    cudaMemcpyAsync(
        this->device_ptrs[0],
        nchw.ptr<float>(),
        nchw.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        this->stream
    );
}

void YOLOv8::Infer()
{
    this->context->enqueueV3(this->stream);

    for (int i = 0; i < this->num_outputs; i++) {
        size_t size = this->output_bindings[i].size * this->output_bindings[i].dsize;

        cudaMemcpyAsync(
            this->host_ptrs[i],
            this->device_ptrs[i + this->num_inputs],
            size,
            cudaMemcpyDeviceToHost,
            this->stream
        );
    }

    cudaStreamSynchronize(this->stream);
}

void YOLOv8::Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}


void YOLOv8::CopyFromMat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->Letterbox(image, nchw, size);
    
    //this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
	nvinfer1::Dims dims{4, {1, 3, size.height, size.width}};
	this->context->setInputShape(this->input_bindings[0].name.c_str(), dims);
    
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs)
{
    char text[256];

    for (auto& obj : objs) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;

        if (y < 0)        y = 0;
        if (y > bgr.rows) y = bgr.rows;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}




