#include <SFML/Graphics.hpp>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include "../vec.h"
#include "../ui.h"
#include "SFML/Window/Keyboard.hpp"

// global rendering parameters
static int screenWidth = 2000;
static int screenHeight = 1125;
static float worldWidth = 20.0f;

sf::RenderWindow window(sf::VideoMode(screenWidth, screenHeight), "gravit simulaton");

// Simulation parameters
const float G       = 6.674e-10f;
const float epsilon = 0.1f;
double pullingStrength = 100.0;
static bool isPaused = false;
static bool isShowingUI = true;

// Random number
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

// Camera variables
Vector2 cameraPos(Vector2(0.0, 0.0));
float moveSpeed = 0.1f;
float zoomSpeed = 0.2f;

// Forward declare
struct QuadTree;

class GravityObject {
    public:
    Vector2 position;
    Vector2 velocity;
    float mass;
    sf::CircleShape shape;
    float radius;
    bool isGrabbed;

    void Draw(sf::RenderWindow &window, Vector2 cameraPos) {
        shape.setPosition(worldToPixel(position, cameraPos, screenWidth, screenHeight, worldWidth));
        shape.setFillColor(sf::Color::White);
        float screenRadius = worldToScreenLength(radius, screenWidth, worldWidth);
        shape.setRadius(screenRadius);
        shape.setOrigin(sf::Vector2f(screenRadius, screenRadius));
        window.draw(shape);
    }

    void UpdateRK4(float dt, const QuadTree& tree, float theta);

    static void ResolveCollisions(std::vector<GravityObject>& gravityObjects) {
        for (size_t i = 0; i < gravityObjects.size(); ++i) {
            for (size_t j = i + 1; j < gravityObjects.size(); ++j) {
                Vector2 delta = gravityObjects[j].position - gravityObjects[i].position;
                float dist = Vector2::length(delta);
                float minDist = gravityObjects[i].radius + gravityObjects[j].radius;

                if (dist < minDist) {
                    Vector2 normal = delta / dist;
                    float overlap = minDist - dist;
                    float totalMass = gravityObjects[i].mass + gravityObjects[j].mass;

                    gravityObjects[i].position -= normal * overlap * (gravityObjects[j].mass / totalMass);
                    gravityObjects[j].position += normal * overlap * (gravityObjects[i].mass / totalMass);

                    Vector2 relVel = gravityObjects[i].velocity - gravityObjects[j].velocity;
                    float dot = Vector2::dot(relVel, normal);

                    if (dot < 0) continue;

                    float impulse = (1.0f * dot) / totalMass;
                    gravityObjects[i].velocity -= normal * impulse * gravityObjects[j].mass;
                    gravityObjects[j].velocity += normal * impulse * gravityObjects[i].mass;
                }
            }
        }
    }

    GravityObject(Vector2 position_, Vector2 velocity_, float mass_, float radius_) {
        position = position_;
        velocity = velocity_;
        mass = mass_;
        radius = radius_;
        isGrabbed = false;
    }
};

struct QuadTree {
    Vector2 center;
    float halfSize;

    float totalMass = 0.f;
    Vector2 centerOfMass = {0.f, 0.f};

    std::unique_ptr<QuadTree> nw, ne, sw, se;
    const GravityObject* body = nullptr;

    bool isLeaf() const { return !nw && !ne && !sw && !se; }

    bool contains(Vector2 pos) const {
        return pos.x >= center.x - halfSize && pos.x < center.x + halfSize &&
               pos.y >= center.y - halfSize && pos.y < center.y + halfSize;
    }

    void subdivide() {
        float q = halfSize * 0.5f;
        nw = std::make_unique<QuadTree>(QuadTree{Vector2(center.x - q, center.y + q), q});
        ne = std::make_unique<QuadTree>(QuadTree{Vector2(center.x + q, center.y + q), q});
        sw = std::make_unique<QuadTree>(QuadTree{Vector2(center.x - q, center.y - q), q});
        se = std::make_unique<QuadTree>(QuadTree{Vector2(center.x + q, center.y - q), q});
    }

    void insert(const GravityObject* obj) {
        if (!contains(obj->position)) return;

        centerOfMass = (centerOfMass * totalMass + obj->position * obj->mass) / (totalMass + obj->mass);
        totalMass += obj->mass;

        if (isLeaf()) {
            if (body == nullptr) {
                body = obj;
                return;
            }
            subdivide();
            const GravityObject* existing = body;
            body = nullptr;
            for (auto* child : {nw.get(), ne.get(), sw.get(), se.get()})
                if (child->contains(existing->position)) { child->insert(existing); break; }
        }

        for (auto* child : {nw.get(), ne.get(), sw.get(), se.get()})
            if (child->contains(obj->position)) { child->insert(obj); break; }
    }

    Vector2 computeAcceleration(const GravityObject* obj, float theta) const {
        if (totalMass == 0.f) return {0.f, 0.f};
        if (isLeaf() && body == obj) return {0.f, 0.f};

        Vector2 delta = centerOfMass - obj->position;
        float dist = Vector2::length(delta);

        if (dist < obj->radius) return {0.f, 0.f};

        float ratio = (halfSize * 2.f) / dist;
        if (isLeaf() || ratio < theta) {
            float forceMag = (G * totalMass) / (dist * dist);
            return delta / dist * forceMag;
        }

        Vector2 acc = {0.f, 0.f};
        for (const auto* child : {nw.get(), ne.get(), sw.get(), se.get()})
            if (child) acc += child->computeAcceleration(obj, theta);
        return acc;
    }
};

// UpdateRK4 defined out-of-line so both types are fully known
void GravityObject::UpdateRK4(float dt, const QuadTree& tree, float theta) {
    if (!isGrabbed) {
        auto computeAcceleration = [&](Vector2 pos) -> Vector2 {
            return tree.computeAcceleration(this, theta);
        };

        Vector2 k1_vel = velocity;
        Vector2 k1_acc = computeAcceleration(position);

        Vector2 k2_vel = velocity + k1_acc * (dt/2);
        Vector2 k2_acc = computeAcceleration(position + k1_vel * (dt/2));

        Vector2 k3_vel = velocity + k2_acc * (dt/2);
        Vector2 k3_acc = computeAcceleration(position + k2_vel * (dt/2));

        Vector2 k4_vel = velocity + k3_acc * dt;
        Vector2 k4_acc = computeAcceleration(position + k3_vel * dt);

        position += (k1_vel + k2_vel * 2 + k3_vel * 2 + k4_vel) * (dt / 6.f);
        velocity += (k1_acc + k2_acc * 2 + k3_acc * 2 + k4_acc) * (dt / 6.f);
    }
    else {
        Vector2 worldMousePosition = pixelToWorld(sf::Vector2f(sf::Mouse::getPosition(window)), cameraPos, screenWidth, screenHeight, worldWidth);
        Vector2 difference = worldMousePosition - position;
        velocity = difference * pullingStrength;
        position += velocity * dt;
    }
}

static std::vector<GravityObject> gravityObjects;

void SpawnGravityObject(std::vector<GravityObject>& gravityObjects, Vector2 pos, Vector2 vel, float mass) {
    float radius = std::cbrt(mass) * 0.0001f;
    gravityObjects.emplace_back(pos, vel, mass, radius);
}

void SpawnRandom(std::vector<GravityObject>& gravityObjects, int n, float boundaryMin, float boundaryMax, float velocityMax, float massMin, float massMax) {
    std::uniform_real_distribution<float> posDist(boundaryMin, boundaryMax);
    std::uniform_real_distribution<float> velDist(-velocityMax, velocityMax);
    std::uniform_real_distribution<float> massDist(massMin, massMax);

    for (int i = 0; i < n; ++i) {
        Vector2 pos(posDist(gen), posDist(gen));
        Vector2 vel(velDist(gen), velDist(gen));
        float mass = massDist(gen);
        SpawnGravityObject(gravityObjects, pos, vel, mass);
    }
}

// UI elements
Background uiBackground(sf::Color(100, 100, 100, 100), sf::Vector2f(10.f, 10.f), sf::Vector2f(1000.f, 1115.f));
CheckBox pausedCheckBox(isPaused, sf::Vector2f(50.0f, 50.0f), 50.0f, sf::Color(100, 100, 100, 200), sf::Color(255, 255, 255, 200));

void CheckToGrabObjects() {
    static bool wasPressed = false;
    bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

    if (isPressed && !wasPressed) {
        for (GravityObject& gravityObj : gravityObjects) {
            sf::Vector2f mousePosition = sf::Vector2f(sf::Mouse::getPosition(window));
            sf::Vector2f difference = gravityObj.shape.getPosition() - mousePosition;
            float screenDistance = std::hypot(difference.x, difference.y);
            if (screenDistance <= gravityObj.shape.getRadius()) {
                gravityObj.isGrabbed = true;
            }
        }
    }
    if (!isPressed) {
        for (GravityObject& gravityObj : gravityObjects) {
            gravityObj.isGrabbed = false;
        }
    }

    wasPressed = isPressed;
}

int main() {
    window.setFramerateLimit(60);

    SpawnRandom(gravityObjects, 20, -8.0f, 8.0f, 2.0f, 1e9f, 1e11f);
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == event.Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space) {
                isPaused = !isPaused;
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Tab) {
                isShowingUI = !isShowingUI;
            }
            if (event.type == sf::Event::MouseButtonPressed) {
                pausedCheckBox.CheckIfPressed(window);
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::E) {
                Vector2 mouseWorld = pixelToWorld(sf::Vector2f(sf::Mouse::getPosition(window)), cameraPos, screenWidth, screenHeight, worldWidth);
                SpawnGravityObject(gravityObjects, mouseWorld, Vector2(0.0, 0.0), 1e10f);
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) cameraPos.x -= moveSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) cameraPos.x += moveSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) cameraPos.y -= moveSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) cameraPos.y += moveSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LBracket)) worldWidth -= zoomSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::RBracket)) worldWidth += zoomSpeed;

        window.clear();

        CheckToGrabObjects();

        if (!isPaused) {
            QuadTree tree;
            tree.center = {0.f, 0.f};
            tree.halfSize = worldWidth * 4.f;
            for (const auto& obj : gravityObjects)
                tree.insert(&obj);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)gravityObjects.size(); ++i)
                gravityObjects[i].UpdateRK4(0.001f, tree, 0.5f);
            for (int iter = 0; iter < 3; ++iter)
                GravityObject::ResolveCollisions(gravityObjects);
        }

        for (GravityObject& obj : gravityObjects)
            obj.Draw(window, cameraPos);

        if (isShowingUI) {
            uiBackground.Draw(window);
            pausedCheckBox.Draw(window);
        }

        window.display();
    }
    return 0;
}
