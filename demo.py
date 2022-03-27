from manim import *
import numpy as np
from image_to_sketch import sketch
from scipy.interpolate import UnivariateSpline
from skimage import feature, io

class demo(ThreeDScene):

    def image_to_parametric(self, file_name, sigma, smoothing, poly_order, neighbor_distance, use_canny, scale):
        # convert image to parametric splines in native scale
        spllist, length, sizex, sizey = sketch(file_name, sigma, smoothing, poly_order, neighbor_distance, use_canny)
        surface = VGroup()
        for i in range(spllist.shape[0]):
            # reconstruct splines from tcks
            spl1x = UnivariateSpline._from_tck(spllist[i, 0])
            spl1y = UnivariateSpline._from_tck(spllist[i, 1])
            # add parametric function rescaled and rotated so that it fits to manim coordinate system
            surface.add(ParametricFunction(
                lambda u: np.array([scale * (spl1y(u) - sizey / 2) /  sizex, -scale * (spl1x(u) - sizex / 2) / sizex, 0]),
                color=WHITE, t_range=[0, 1]))

        return surface, length

    def final(self):
        obj1, llength1 = self.image_to_parametric("beaker.png", 2, 100, 3, 2, True, 4)
        obj2, llength2 = self.image_to_parametric("tube.png", 2, 100, 3, 3, True, 2)
        obj3, llength3 = self.image_to_parametric("puff.jpg", 2, 100, 3, 2, True, 0.1)
        obj4 = obj2.copy()
        llengthsum1 = np.sum(llength1)
        llengthsum2 = np.sum(llength2)
        llengthsum3 = np.sum(llength3)
        obj1.set_stroke(width=2)
        obj2.set_stroke(width=2)
        obj3.set_stroke(width=2)
        obj4.set_stroke(width=2)
        obj1.set_color(BLACK)
        obj2.set_color(BLACK)
        obj3.set_color(BLACK)
        obj4.set_color(BLACK)
        obj1.set_x(0)
        obj1.set_y(-2)
        obj2.set_x(-1.5)
        obj2.set_y(1)
        obj4.flip()
        obj4.set_x(1.5)
        obj4.set_y(1)
        obj3.set_x(0)
        obj3.set_y(0)
        total_runtime=1.5
        img1 = ImageMobject("bkg.jpg")
        self.add(img1)

        for counter, g in enumerate(obj1):
            if (total_runtime * llength1[counter] / llengthsum1)<0.0067:
                self.add(g)
            else:
                self.play(Create(g), run_time=total_runtime * llength1[counter] / llengthsum1)
        self.wait(0.5)

        for i in range(0, len(obj2)):
            self.play(Create(obj2[i]), Create(obj4[i]), run_time=total_runtime * llength2[i] / llengthsum2)

        self.wait(1)
        self.play(ApplyMethod(obj1.scale, 0), ApplyMethod(obj2.scale, 0), ApplyMethod(obj4.scale, 0), FadeIn(obj3), run_time=1)
        self.play(ApplyMethod(obj3.scale, 100), run_time=1.5)
        self.play(FadeOut(obj3), run_time=0.5)
        return 0

    def construct(self):
        axes = ThreeDAxes()
        image1 = io.imread("bottle.jpg", as_gray=True)
        image1_canny = feature.canny(image1, sigma=1, mode='constant', cval = False)

        io.imsave("bottle_canny.jpg", image1_canny)
        obj1, llength = self.image_to_parametric("bottle.jpg", 1, 1000, 3, 3, True, 3)
        obj2, llength2 = self.image_to_parametric("bottle.jpg", 1, 5000, 2, 3, True, 3)
        text_title = VGroup(Text("Image to parametric curve sketch"), Text("by Edi Topić")).arrange(DOWN, center=True)
        text_title.font_size = 30
        self.play(Write(text_title))
        self.wait(1)
        self.play(Unwrite(text_title))

        image1_mobject = ImageMobject("bottle.jpg")
        image1_mobject.set_x(-4)

        image1_canny_mobject = ImageMobject("bottle_canny.jpg")
        image1_canny_mobject.set_x(0)

        obj1.set_x(4)
        obj1.set_stroke(width=1)
        total_runtime = 3

        obj2.set_x(4)
        obj2.set_stroke(width=1)
        obj2.set_color(GOLD)

        llengthsum = np.sum(llength)

        self.play(FadeIn(image1_mobject))

        self.play(FadeIn(image1_canny_mobject))

        text2 = VGroup(Text("Canny filter"), Text("sigma = 1")).arrange(DOWN, center=True).set_y(-3)
        for g in text2: g.font_size = 20

        self.play(FadeIn(text2))
        for counter, g in enumerate(obj1):
            if (total_runtime * llength[counter] / llengthsum)<0.0067:
                self.add(g)
            else:
                self.play(Create(g), run_time=total_runtime * llength[counter] / llengthsum)
        text3 = VGroup(Text("Sketch, cubic spline"), Text("smoothing = 1000")).arrange(DOWN, center=True).set_x(4).set_y(-3)
        text4 = VGroup(Text("Sketch, quad spline"), Text("smoothing = 100")).arrange(DOWN, center=True).set_x(
            4).set_y(-3)
        for g in text3: g.font_size = 20
        for g in text4: g.font_size = 20
        self.play(FadeIn(text3))
        self.wait(2)
        self.play(ReplacementTransform(obj1, obj2), FadeTransform(text3, text4))
        self.wait(2)
        self.remove(image1_mobject, image1_canny_mobject, *obj1, *obj2, *text2, *text3, *text4)
        self.wait(1)

        im2 = ImageMobject("cham.jpg")
        im2.set_x(-4)

        obj3, llength3 = self.image_to_parametric("cham.jpg", 1, 1000, 3, 4, True, 4)
        obj4, llength4 = self.image_to_parametric("cham.jpg", 2, 1000, 3, 4, True, 4)
        obj5, llength5 = self.image_to_parametric("cham.jpg", 3, 1000, 3, 4, True, 4)
        obj6, llength6 = self.image_to_parametric("cham.jpg", 5, 1000, 3, 4, True, 4)
        obj3.set_stroke(width=1)
        obj4.set_stroke(width=1)
        obj5.set_stroke(width=1)
        obj6.set_stroke(width=1)
        obj3.set_x(4)
        obj4.set_x(4)
        obj5.set_x(4)
        obj6.set_x(4)
        textch1 = Text("sigma=1")
        textch2 = Text("sigma=2")
        textch3 = Text("sigma=3")
        textch4 = Text("sigma=5")
        self.play(FadeIn(im2))
        self.play(Create(obj3), run_time=1)
        self.add(textch1)
        self.wait(2)
        self.remove(*obj3, textch1)
        self.play(Create(obj4), run_time=1)
        self.add(textch2)
        self.wait(2)
        self.remove(*obj4,textch2)
        self.play(Create(obj5), run_time=1)
        self.add(textch3)
        self.wait(2)
        self.remove(*obj5,textch3)
        self.play(Create(obj6), run_time=1)
        self.add(textch4)
        self.wait(2)
        self.remove(*obj6, textch4, im2)
        self.play(FadeIn(Text("Animation example")))
        self.wait(2)
        self.final()











