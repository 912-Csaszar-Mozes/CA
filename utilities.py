import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Imaging:
    def __init__(self):
        self.first_draw = True
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def show(self, draw_func, *args):
        if not self.first_draw:
            self.ax.cla()

        draw_func(*args)
        if self.first_draw:
            plt.show(block=False)
            self.first_draw = False
        else:
            plt.draw()
        plt.pause(0.1)


    def show_image(self, tensor, title='Plain Image'):
        plt.clf()
        
        numpy_img = tensor.permute(1, 2, 0).numpy()  # Change tensor shape to (height, width, channels) for plotting
        
        # Plot the image using matplotlib
        plt.imshow(numpy_img)
        plt.axis('off')  # Hide axis values
        plt.title(title)  # Set title if needed
        plt.draw()

    def _show_image_boxes(self, tensor, dets, title):
        numpy_img = tensor.permute(1, 2, 0).numpy()  # Change tensor shape to (height, width, channels) for plotting

        self.ax.imshow(numpy_img)
        self.ax.set_title(title)
    
        for det in dets:
            left_bottom = (det[0], det[1])
            width, height = (det[2] - det[0], det[3] - det[1])
            rect = patches.Rectangle(left_bottom, width, height, linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)

    def show_image_boxes(self, tensor, dets, title='Boxes'):
        self.show(self._show_image_boxes, tensor, dets, title)

    def show_image_boxes_faces(self, tensor, dets, faces, face_subplots=(10,10)):
        plt.clf()
        
        numpy_img = tensor.permute(1, 2, 0).numpy()  # Change tensor shape to (height, width, channels) for plotting
        plt.axis('off')
    
        fig = plt.figure(figsize=(18.4,6))
        subfigs = fig.subfigures(3, 1)
        
        for i, subfig in enumerate(subfigs.flat):
            # plot original image
            if i == 0:
                ax = subfig.subplots(1,1)
                ax.set_title('Original')
                ax.axis('off')
                ax.imshow(numpy_img)
            # plot boxes on the image
            elif i == 1:
                ax = subfig.subplots(1,1)
                ax.set_title('Boxes on Image')
                ax.imshow(numpy_img)
                ax.axis('off')
    
                for det in dets:
                    left_bottom = (det[0], det[1])
                    width, height = (det[2] - det[0], det[3] - det[1])
                    rect = patches.Rectangle(left_bottom, width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            # show faces only
            else:
                subfig.suptitle('Faces')  # Set title if needed
                
                axs = subfig.subplots(*face_subplots)
            
                for j in range(len(faces)):
                    face_img = faces[j].permute(1, 2, 0).numpy()        
                    axs[j // face_subplots[1]][j % face_subplots[1]].imshow(face_img)
                    axs[j // face_subplots[1]][j % face_subplots[1]].axis('off')
                
        plt.draw()

    def show_faces(self, faces, subplots=(10,10)):
        plt.clf()
        
        fig, axs = plt.subplots(subplots[0], subplots[1])
        for i in range(len(faces)):
            face_img = faces[i].permute(1, 2, 0).numpy()
            axs[i//subplots[1]][i % subplots[1]].imshow(face_img)
        plt.axis('off')  # Hide axis values
        plt.title('Faces')  # Set title if needed
        
        plt.draw()