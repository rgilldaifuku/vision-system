# What this project is (plain English)

This guide is for anyone who wants to understand the project **without** reading code or technical manuals. If you need step-by-step technical commands, use **`README.md`** in the same folder.

---

## The big idea in one sentence

This project teaches a computer to **recognize a specific thing in photos or live camera video**—for example a treat called “yellow daifuku”—and then **show you when it sees it**, with optional extras that can move a small motor (servo) when used with extra hardware.

Think of it like teaching someone to spot Waldo in a crowd: you show them many pictures with Waldo circled, they practice, then they can point him out in new pictures.

---

## What “teaching the computer” means here

Computers do not “understand” images the way people do. This project uses a **trained model**—a large pattern file the computer builds from examples.

1. **You provide photos** of the real world (your webcam or saved pictures).
2. **You (or a tool) mark** where the object is in each photo—usually a rectangle around it. Those marks are stored as companion files next to the images.
3. **A training step** runs for a long time while the computer adjusts internal numbers so that, on average, it gets better at matching “this kind of image” to “object here.”
4. **When you’re done**, you get a **result file** (often described as “weights”) that the live program loads so it can recognize the object in new video.

Training can take a while and may make the computer fan run; that is normal.

---

## What you actually see when it’s working

When someone runs the main **desktop window** program:

- The **camera feed** appears like a video call.
- When the system thinks it sees something it was trained on, it draws **boxes and labels** on the video.
- There is a **confidence** control: higher means “only tell me when you’re more sure”; lower means “tell me more often, but I might get more false alarms.”
- There is a simple **“found it” / “didn’t find it”** style indicator. In the example setup, it lights up when the trained label matches something like “yellow daifuku.” If you train on something else, whoever maintains the project may need to update that wording in the program.

So for a non-coder: **the product is mainly “smart webcam + on-screen boxes and a yes/no light,”** not a website or a phone app (unless someone wraps it that way later).

---

## The optional “motor” part (only if you use that hardware)

Some people connect a **small rotating motor (servo)** to a board like an **Arduino** (a tiny computer that talks over USB).

Then a different script can say, in effect: “when you see something in the camera, send signals to the Arduino,” and the Arduino moves the motor within safe limits.

You do **not** need that hardware to understand the core idea. It is an add-on for physical demos or projects.

---

## What the main folders and files “mean” in everyday terms

| Everyday description | Where it usually lives |
|----------------------|-------------------------|
| **Photo homework** — pictures used for learning | `datasets/my_items` (and subfolders for “practice” vs “quiz” sets) |
| **The lesson plan** — tells the training step what the object is called and where folders are | `datasets/my_items/data.yaml` |
| **“Teach the model now”** button (run by someone who uses the terminal) | `train_my_items.py` |
| **“Show me the live camera with boxes”** window | `new_app.py` |
| **“Camera + optional motor signals”** (more technical setup) | `servo_tracker.py` and `servo_tracker_arduino.ino` |
| **“Shuffle some practice photos into the quiz pile”** helper | `make_val_from_train.py` |
| **“Install the helper software this project needs”** | `dependencies.py` |
| **Graduation certificates** — outputs from training (the files you load for live detection) | `runs` (inside there are folders with names like training runs) |

Names with `.py` are **Python** programs—small automation files developers run from a command line. You do not have to open them if someone else runs the project for you.

---

## Who does what (if you are not the person at the keyboard)

- **You** might collect photos, decide what object matters, and try the live window to see if it behaves well enough for your use.
- **Someone comfortable with computers** installs Python, runs the dependency installer, runs training, points the live app at the correct result file, and fixes paths if the project was moved to another PC.
- **If something is wrong**, it is often one of these: wrong camera selected, training never finished, the “result file” from training is not the one the live app is using, or photos/labels are not in the expected folders.

---

## Limits (honest expectations)

- It only works well on things **similar to what it was trained on** (lighting, angle, distance, background).
- It can be **wrong** sometimes (misses the object or thinks it sees it when it does not). The confidence slider changes how picky it is, not perfection.
- It is a **local** project on a computer: it is not automatically “in the cloud” or an official product unless you set that up separately.

---

## Where to go next

- **Technical setup, commands, and troubleshooting:** open **`README.md`** in this same folder.
- **Questions for your technical helper:** “Is Python installed?” “Did training finish and where is `best.pt`?” “Which camera index are we using?” They will know what those mean.

If this guide ever disagrees with the actual programs, **the programs win**—this file is only meant to explain the intent in simple language.
