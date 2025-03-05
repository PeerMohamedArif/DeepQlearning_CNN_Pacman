import gymnasium as gym
import cv2

def show_video_of_model(agent, env_name='MsPacmanDeterministic-v0', output_filename='output_video.mp4'):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False

    frame = env.render()
    height, width, layers = frame.shape

    video_writer = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (width, height)
    )

    try:
        while not done:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
    finally:
        env.close()
        video_writer.release()

    print(f"Video saved as {output_filename}")



# for jupyternotebook

# import glob
# import io
# import base64
# import imageio
# from IPython.display import HTML, display
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# def show_video_of_model(agent, env_name):
#     env = gym.make(env_name, render_mode='rgb_array')
#     state, _ = env.reset()
#     done = False
#     frames = []
#     while not done:
#         frame = env.render()
#         frames.append(frame)
#         action = agent.act(state)
#         state, reward, done, _, _ = env.step(action)
#     env.close()
#     imageio.mimsave('video.mp4', frames, fps=30)

# show_video_of_model(agent, 'MsPacmanDeterministic-v0')

# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# show_video()
