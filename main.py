import streamlit as st
import cv2
import numpy as np
import SimpleITK as sitk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd

# تعريف المتغيرات العالمية هنا
global merged_image
global registered_second_image
global start_point, end_point
start_point = None
end_point = None
results = []
def preview_merged_image():
    global merged_image
    if merged_image is None:
        print("No merged image available.")
        return
    merged_image_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(merged_image_rgb)
    image = ImageTk.PhotoImage(image)
    top = Toplevel()
    top.title("Merged Image Preview")
    top.columnconfigure(0, weight=1)
    top.rowconfigure(0, weight=1)
    canvas = Canvas(top, width=image.width(), height=image.height())
    canvas.grid(row=0, column=0)
    canvas.create_image(0, 0, anchor=NW, image=image)
    canvas.image = image

    def on_mouse_click(event):
        global start_point, end_point
        x, y = event.x, event.y
        if start_point is None:
            start_point = (x, y)
        else:
            end_point = (x, y)
            calculate_distance()
            # رسم خط بين النقطتين على الصورة
            canvas.create_line(start_point, end_point, fill="red", width=2)
            start_point = None  # إعادة تعيين النقطتين بعد الحساب

    canvas.bind("<Button-1>", on_mouse_click)

# تعريف الوظائف الأخرى هنا (calculate_distance وغيرها)
def calculate_distance():
    global start_point, end_point, results

    if start_point is None or end_point is None:
        return

    x1, y1 = start_point
    x2, y2 = end_point

    pixel_to_mm_conversion_factor = 0.1
    distance_in_mm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * pixel_to_mm_conversion_factor

    if distance_in_mm == 1.00:
        result = "100%"
    elif 0.8 <= distance_in_mm <= 1.2:
        result = "50%"
    else:
        result = "0%"
    if distance_in_mm == 2.0:
        result = "100%"
    elif 1.0 > distance_in_mm <= 3.0:
        result = "50%"
    data = {
        'Distance (mm)': distance_in_mm,
        'Result': result,
        #'Success Rate': selected_rate
    }

    results.append(data)  # Append the calculated data to the results list

    # Create a DataFrame from the results list
    df = pd.DataFrame(results)

    # حفظ البيانات في ملف Excel
    df.to_excel('measurements.xlsx', index=False)

    distance_window = Toplevel()
    distance_window.title("Distance Measurement")
    distance_window.geometry("400x150")

    success_rate_label = Label(distance_window, text="Enter Success Rate (0.0 to 21.0):")
    success_rate_label.pack()
    success_rate_entry = Entry(distance_window)
    success_rate_entry.pack()
    Label(distance_window, text=f"Distance between points: {distance_in_mm:.2f} mm").pack()

    def calculate_result():
        success_rate_text = success_rate_entry.get()

        try:
            selected_rate = float(success_rate_text)
        except ValueError:
            result_label.config(text="Invalid input. Enter a number between 0.0 and 21.0.")
            return

        if 0.0 <= selected_rate <= 21.0:
            result_label.config(text=f"Selected Success Rate: {selected_rate:.1f}\nResult: {result}")
        else:
            result_label.config(text="Invalid input. Enter a number between 0.0 and 21.0.")

    calculate_button = Button(distance_window, text="Calculate Result", command=calculate_result)
    calculate_button.pack()

    result_label = Label(distance_window, text="")
    result_label.pack()

def open_first_image():
    global first_image
    file_path = filedialog.askopenfilename()
    first_image = cv2.imread(file_path)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    display_image(first_image, image_canvas1)


def open_second_image():
    global second_image, first_image, merged_image, registered_second_image
    file_path = filedialog.askopenfilename()
    second_image = cv2.imread(file_path)
    second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
    display_image(second_image, image_canvas2)
    if first_image is not None:
        if registration_method_var.get() == "SIFT":
            registered_second_image = register_images_sift(second_image, first_image)
        else:
            registered_second_image = register_images_simpleitk(first_image, second_image)
        display_image(registered_second_image, image_canvas4)
    update_comparison()
    merged_image = merge_images(first_image, registered_second_image)
    display_image(merged_image, image_canvas5)


def update_comparison():
    global registered_second_image
    if first_image is not None and second_image is not None:
        threshold_value = threshold_slider.get()
        difference_image = compute_difference_image(first_image, registered_second_image, threshold_value)
        display_image(difference_image, image_canvas3)


def display_image(image, canvas):
    image = Image.fromarray(image)
    image.thumbnail((300, 300))
    image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=NW, image=image)
    canvas.image = image


def compute_difference_image(img1, img2, threshold_value):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(img1_gray, img2_gray)
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    _, diff_thresholded = cv2.threshold(blurred_diff, threshold_value, 255, cv2.THRESH_BINARY)
    diff_thresholded_color = cv2.cvtColor(diff_thresholded, cv2.COLOR_GRAY2RGB)

    # Calculate the size of the difference in millimeters

    return diff_thresholded_color


def compare_images():
    update_comparison()


def register_images_sift(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width, _ = img2.shape
    registered_image = cv2.warpPerspective(img1, M, (width, height))

    return registered_image


def register_images_simpleitk(img1, img2, transform_type="affine"):
    if transform_type == "affine":
        img1_sitk = sitk.GetImageFromArray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        img2_sitk = sitk.GetImageFromArray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))

        registration_method = sitk.ImageRegistrationMethod()

        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        final_transform = sitk.AffineTransform(2)
        registration_method.SetInitialTransform(final_transform, inPlace=False)

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        final_transform = registration_method.Execute(sitk.Cast(img1_sitk, sitk.sitkFloat32),
                                                      sitk.Cast(img2_sitk, sitk.sitkFloat32))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img1_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)

        registered_img2_sitk = resampler.Execute(img2_sitk)
        registered_img2 = cv2.cvtColor(sitk.GetArrayFromImage(registered_img2_sitk), cv2.COLOR_GRAY2RGB)

        return registered_img2
    else:  # transform_type == "bspline"
        img1_sitk = sitk.GetImageFromArray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        img2_sitk = sitk.GetImageFromArray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))

        registration_method = sitk.ImageRegistrationMethod()

        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        grid_physical_spacing = [50.0, 50.0]
        transform_domain_mesh_size = [4, 4]

        initial_transform = sitk.BSplineTransformInitializer(img1_sitk, transform_domain_mesh_size, order=3)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        final_transform = registration_method.Execute(sitk.Cast(img1_sitk, sitk.sitkFloat32),
                                                      sitk.Cast(img2_sitk, sitk.sitkFloat32))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img1_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)

        registered_img2_sitk = resampler.Execute(img2_sitk)
        registered_img2 = cv2.cvtColor(sitk.GetArrayFromImage(registered_img2_sitk), cv2.COLOR_GRAY2RGB)

        return registered_img2


def merge_images(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)

    img1_green = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    img1_green[:, :, 0] = 0
    img1_green[:, :, 2] = 0

    img2_purple = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2RGB)
    img2_purple[:, :, 1] = 0

    merged_image = cv2.addWeighted(img1_green, 0.5, img2_purple, 0.5, 0)
    display_image(merged_image, image_canvas5)

    return merged_image


def register_images_wrapper(event=None):
    global registered_second_image
    if first_image is not None and second_image is not None:
        method = registration_method_var.get()
        if method == "SIFT":
            registered_second_image = register_images_sift(second_image, first_image)
        elif method == "Affine":
            registered_second_image = register_images_simpleitk(first_image, second_image, "affine")
        else:  # method == "B-spline"
            registered_second_image = register_images_simpleitk(first_image, second_image, "bspline")
        display_image(registered_second_image, image_canvas4)
        update_comparison()
        merged_image = merge_images(first_image, registered_second_image)
        display_image(merged_image, image_canvas5)



logo_image = Image.open("C:\\Users\\DELL\\OneDrive\\Desktop\\dent.jpg")

st.image(logo_image, use_column_width=True, caption="Comparison App")
st.title("Image Comparison")


uploaded_first_image = st.file_uploader("Upload the First Image", type=["jpg", "png", "jpeg"])
uploaded_second_image = st.file_uploader("Upload the Second Image", type=["jpg", "png", "jpeg"])

if uploaded_first_image is not None and uploaded_second_image is not None:
                first_image = Image.open(uploaded_first_image)
                st.image(first_image, caption="First Image", use_column_width=True)
                second_image = Image.open(uploaded_second_image)
                st.image(second_image, caption="Second Image", use_column_width=True)



if st.button("Compare and Register Images"):

            root = Tk()
            root.title("Image Comparison And registration")
            root.configure(bg="#9370DB")
            first_image = None
            second_image = None

            root.columnconfigure(0, weight=1)
            root.columnconfigure(1, weight=1)
            root.rowconfigure(0, weight=1)
            root.rowconfigure(1, weight=1)
            root.rowconfigure(2, weight=1)

            Label(root, text="Registration Method").grid(row=0, column=2, pady=5, padx=5, sticky="w")
            registration_method_var = StringVar()
            registration_method_var.set("SIFT")
            registration_method_menu = ttk.Combobox(root, textvariable=registration_method_var,
                                                    values=["SIFT", "Affine", "B-spline"], state='readonly', width=10)
            registration_method_menu.grid(row=0, column=2, pady=5, padx=5, sticky="e")
            registration_method_menu.bind("<<ComboboxSelected>>", register_images_wrapper)

            Label(root, text="First Image").grid(row=0, column=0, pady=5, padx=5, sticky="w")
            Label(root, text="Second Image").grid(row=0, column=1, pady=5, padx=5, sticky="w")
            Label(root, text="Registered Second Image").grid(row=0, column=2, pady=5, padx=5, sticky="w")
            Label(root, text="Difference Image").grid(row=3, column=1, pady=5, padx=5, sticky="w")

            Button(root, text="First Picture", command=open_first_image).grid(row=0, column=0, pady=5, padx=5,
                                                                              sticky="w")
            image_canvas1 = Canvas(root, width=300, height=300)
            image_canvas1.grid(row=1, column=0)

            Button(root, text="Second Picture", command=open_second_image).grid(row=0, column=1, pady=5, padx=5,
                                                                                sticky="w")
            image_canvas2 = Canvas(root, width=300, height=300)
            image_canvas2.grid(row=1, column=1)

            image_canvas4 = Canvas(root, width=300, height=300)
            image_canvas4.grid(row=1, column=2)

            threshold_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Threshold",
                                     command=lambda value: update_comparison())
            threshold_slider.set(30)
            threshold_slider.grid(row=2, column=0, columnspan=3, sticky="ew")

            image_canvas3 = Canvas(root, width=300, height=300)
            image_canvas3.grid(row=4, column=1)

            Label(root, text="Merged Image").grid(row=3, column=2, pady=5, padx=5, sticky="w")
            image_canvas5 = Canvas(root, width=300, height=300)
            image_canvas5.grid(row=4, column=2)
            Button(root, text="Preview", command=preview_merged_image).grid(row=5, column=2, pady=5, padx=5, sticky="w")

            root.mainloop()