#include <SFML/Graphics.hpp>
#include <memory>
#include <sstream>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include "../vec.h"
#include "../ui.h"
#include "SFML/Window/Keyboard.hpp"

// global rendering parameters
static float worldWidth = 20.0f;

sf::RenderWindow window(sf::VideoMode::getDesktopMode(), "gravit simulaton");
static int screenWidth = window.getSize().x;
static int screenHeight = window.getSize().y;

// Simulation parameters
double G       = 40.0;
const float epsilon = 0.1f;
double pullingStrength = 100.0;
bool isPaused = false;
bool isShowingUI = true;
bool doGrabCarefully = false;
double grabRadius = 3.0;
double simulationSpeed = 1.0;

// Random number
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

// Camera variables
Vector2 cameraPos(Vector2(0.0, 0.0));
float moveSpeed = 0.005f;
float zoomSpeed = 0.010f;

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

    Vector2 computeAcceleration(Vector2 queryPos, float objRadius, float theta) const {
        if (totalMass == 0.f) return {0.f, 0.f};

        Vector2 delta = centerOfMass - queryPos;
        float dist = Vector2::length(delta);

        if (dist < objRadius) return {0.f, 0.f};

        float ratio = (halfSize * 2.f) / dist;
        if (isLeaf() || ratio < theta) {
            float forceMag = (G * totalMass) / (dist * dist);
            return delta / dist * forceMag;
        }

        Vector2 acc = {0.f, 0.f};
        for (const auto* child : {nw.get(), ne.get(), sw.get(), se.get()})
            if (child) acc += child->computeAcceleration(queryPos, objRadius, theta);
        return acc;
    }

    double computePotentialEnergy(const GravityObject* obj, float theta) const {
        if (totalMass == 0.f) return 0.0;
        if (isLeaf() && body == obj) return 0.0;

        Vector2 delta = centerOfMass - obj->position;
        float dist = Vector2::length(delta);

        if (dist < obj->radius) return 0.0;

        float ratio = (halfSize * 2.f) / dist;
        if (isLeaf() || ratio < theta) {
            return -0.5 * (G * obj->mass * totalMass) / dist;
        }

        double pe = 0.0;
        for (const auto* child : {nw.get(), ne.get(), sw.get(), se.get()})
            if (child) pe += child->computePotentialEnergy(obj, theta);
        return pe;
    }
};

// UpdateRK4 defined out-of-line so both types are fully known
void GravityObject::UpdateRK4(float dt, const QuadTree& tree, float theta) {
    if (!isGrabbed) {
        auto accel = [&](Vector2 pos) -> Vector2 {
            return tree.computeAcceleration(pos, radius, theta);
        };

        Vector2 k1_v = velocity;
        Vector2 k1_a = accel(position);

        Vector2 k2_v = velocity + k1_a * (dt / 2);
        Vector2 k2_a = accel(position + k1_v * (dt / 2));

        Vector2 k3_v = velocity + k2_a * (dt / 2);
        Vector2 k3_a = accel(position + k2_v * (dt / 2));

        Vector2 k4_v = velocity + k3_a * dt;
        Vector2 k4_a = accel(position + k3_v * dt);

        position += (k1_v + k2_v * 2 + k3_v * 2 + k4_v) * (dt / 6.f);
        velocity += (k1_a + k2_a * 2 + k3_a * 2 + k4_a) * (dt / 6.f);
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
    float radius = std::cbrt(mass) * 0.2f;
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

void SpawnGalaxy(std::vector<GravityObject>& gravityObjects, int n, float radius, float massMin, float massMax, Vector2 center = {0,0}, Vector2 bulkVelocity = {0,0}) {
    // Central black hole
    float blackHoleMass = 500.f;
    SpawnGravityObject(gravityObjects, center, bulkVelocity, blackHoleMass);

    std::uniform_real_distribution<float> angleDist(0.f, 2.f * M_PI);
    std::exponential_distribution<float> radDist(3.0f / radius);
    std::uniform_real_distribution<float> massDist(massMin, massMax);
    std::normal_distribution<float> thicknessDist(0.f, radius * 0.04f);

    for (int i = 0; i < n; ++i) {
        float angle = angleDist(gen);
        float r = std::min(radDist(gen), radius);

        Vector2 pos = center + Vector2(std::cos(angle) * r, std::sin(angle) * r);
        pos.y += thicknessDist(gen);

        float enclosedMass = blackHoleMass;
        float speed = std::sqrt((float)G * enclosedMass / (r + 0.1f));

        // Add slight velocity dispersion
        std::normal_distribution<float> jitter(0.f, speed * 0.05f);
        Vector2 tangent(-std::sin(angle), std::cos(angle));
        Vector2 vel = bulkVelocity + tangent * speed + Vector2(jitter(gen), jitter(gen));

        float mass = massDist(gen);
        SpawnGravityObject(gravityObjects, pos, vel, mass);
    }
}

void CheckToGrabObjects() {
    static bool wasPressed = false;
    bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

    if (isPressed && !wasPressed) {
        for (GravityObject& gravityObj : gravityObjects) {         
            sf::Vector2f mousePosition = sf::Vector2f(sf::Mouse::getPosition(window));
            sf::Vector2f difference = gravityObj.shape.getPosition() - mousePosition;
            float screenDistance = std::hypot(difference.x, difference.y);
            if (doGrabCarefully) {
                if (screenDistance <= gravityObj.shape.getRadius()) {
                    gravityObj.isGrabbed = true;
                }
            }
            else {
                if (screenToWorldLength(screenDistance, screenWidth, 20.0f) <= grabRadius) {
                    gravityObj.isGrabbed = true;
                }
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

void ApplySwirlForce(std::vector<GravityObject>& gravityObjects, Vector2 cursorWorld, double strength, double radius, float dt) {
    for (GravityObject& obj : gravityObjects) {
        Vector2 delta = obj.position - cursorWorld;
        float dist = Vector2::length(delta);

        if (dist > radius || dist < 0.0001f) continue;

        Vector2 tangent(-delta.y, delta.x);
        tangent = tangent / dist;

        float falloff = std::sqrt((worldWidth / 10.0f) / (dist + epsilon));

        obj.velocity += tangent * (float)(strength * falloff * dt);
    }
}

double ComputeKineticEnergy(const std::vector<GravityObject>& objs) {
    double ke = 0.0;
    for (const auto& obj : objs) {
        double v2 = obj.velocity.x * obj.velocity.x + obj.velocity.y * obj.velocity.y;
        ke += 0.5 * obj.mass * v2;
    }
    return ke;
}

double ComputePotentialEnergy(const std::vector<GravityObject>& objs, const QuadTree& tree, float theta) {
    double pe = 0.0;
    for (const auto& obj : objs)
        pe += tree.computePotentialEnergy(&obj, theta);
    return pe;
}

#include "gravity_gpu.h"

int main() {
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.loadFromFile("font.tff")) {
        return -1;
    }

    // Initialise GPU solver
    GravityGPU gpu;

    sf::CircleShape grabRadiusCircle(0, 60);
    grabRadiusCircle.setFillColor(sf::Color(255, 255, 255, 30));

    // UI elements
    Background uiBackground(sf::Color(100, 100, 100, 100),
                            sf::Vector2f(10.f, 10.f),
                            sf::Vector2f(500.f, window.getSize().y - 10.f)
                        );
    CheckBox pausedCheckBox(isPaused,
                            sf::Vector2f(50.0f, 50.0f),
                            50.0f,
                            sf::Color(100, 100, 100, 200),
                            sf::Color(255, 255, 255, 200),
                            font,
                            "paused",
                            20
                        );
    CheckBox grabModeCheckBox(doGrabCarefully,
                              sf::Vector2f(50.0f, 120.0f),
                              50.0f,
                              sf::Color(100, 100, 100, 200),
                              sf::Color(255, 255, 255, 200),
                              font,
                              "precise grabbing",
                              20
                          );
    Slider grabRadiusSlider(grabRadius,
                            0.1,
                            10.0,
                            sf::Vector2f(50.0f, 250.0f),
                            400.0f,
                            sf::Color(100, 100, 100, 255),
                            sf::Color::White,
                            5.f,
                            10.f,
                            2,
                            font,
                            "grabbing radius",
                            20
                        );
    Slider GSlider(G,
                            -20.0,
                            40.0,
                            sf::Vector2f(50.0f, 400.0f),
                            400.0f,
                            sf::Color(100, 100, 100, 255),
                            sf::Color::White,
                            5.f,
                            10.f,
                            2,
                            font,
                            "gravitational constant",
                            20
                        );
    Slider speedSlider(simulationSpeed,
                            0.0,
                            1.0,
                            sf::Vector2f(50.0f, 550.0f),
                            400.0f,
                            sf::Color(100, 100, 100, 255),
                            sf::Color::White,
                            5.f,
                            10.f,
                            2,
                            font,
                            "simulation speed",
                            20
                        );
    Text text1(font, "'E': Spawn object\n'R': Delete all objects\n'G': Spawn galaxy\n'T': Zero velocities\n'[' / ']': Zoom in/out\n'W', 'A', 'S', 'D': Move the camera", sf::Vector2f(50.0f, 600.f), 20, sf::Color::White);
    std::ostringstream energyStream;
    Text energyText(font, "", sf::Vector2f(50.0f, 800.f), 18, sf::Color::White);
    sf::Vector2f mousePosition;

    // Start with a galaxy instead of random objects
    SpawnGalaxy(gravityObjects, 10000, 1000.0f, 0.5f, 2.0f);


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
                grabModeCheckBox.CheckIfPressed(window);
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::E) {
                    Vector2 mouseWorld = pixelToWorld(sf::Vector2f(sf::Mouse::getPosition(window)), cameraPos, screenWidth, screenHeight, worldWidth);
                    SpawnGravityObject(gravityObjects, mouseWorld, Vector2(0.0, 0.0), 1.f);
                }
                if (event.key.code == sf::Keyboard::R) {
                    gravityObjects.clear();
                }
                if (event.key.code == sf::Keyboard::G) {
                    // Spawn a galaxy at mouse position
                    Vector2 mouseWorld = pixelToWorld(sf::Vector2f(sf::Mouse::getPosition(window)), cameraPos, screenWidth, screenHeight, worldWidth);
                    SpawnGalaxy(gravityObjects, 200, 7.0f, 0.1f, 1.0f, mouseWorld);
                }
                if (event.key.code == sf::Keyboard::T) {
                    for (GravityObject& obj : gravityObjects) {
                        obj.velocity = Vector2(0.0, 0.0);
                    }
                }
            }
            grabRadiusSlider.HandleEvent(event, window);
            GSlider.HandleEvent(event, window);
            speedSlider.HandleEvent(event, window);
        }

        mousePosition.x = (float)sf::Mouse::getPosition(window).x;
        mousePosition.y = (float)sf::Mouse::getPosition(window).y;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) cameraPos.x -= moveSpeed*worldWidth;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) cameraPos.x += moveSpeed*worldWidth;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) cameraPos.y -= moveSpeed*worldWidth;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) cameraPos.y += moveSpeed*worldWidth;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LBracket)) worldWidth -= zoomSpeed*worldWidth;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::RBracket)) worldWidth += zoomSpeed*worldWidth;

        window.clear();

        CheckToGrabObjects();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::U) && !isPaused) {
            Vector2 cursorWorld = pixelToWorld(
                sf::Vector2f(sf::Mouse::getPosition(window)),
                cameraPos, screenWidth, screenHeight, worldWidth
            );
            ApplySwirlForce(gravityObjects, cursorWorld, 500.0, grabRadius * 3.0, 0.001f * simulationSpeed);
        }

        // QuadTree still used for potential energy display
        QuadTree tree;
        tree.center = {0.f, 0.f};
        tree.halfSize = worldWidth * 4.f;
        for (const auto& obj : gravityObjects)
            tree.insert(&obj);

        if (!isPaused) {
            // GPU handles force integration
            gpu.step(gravityObjects, 0.001f * simulationSpeed, (float)G, epsilon);
            for (int iter = 0; iter < 8; ++iter)
                GravityObject::ResolveCollisions(gravityObjects);
        }

        for (GravityObject& obj : gravityObjects) {
            obj.Draw(window, cameraPos);
            if (obj.isGrabbed) {
                obj.shape.setFillColor(sf::Color::Red);
            }
            else {
                obj.shape.setFillColor(sf::Color::White);
            }
        }

        if (isShowingUI) {
            uiBackground.Draw(window);
            pausedCheckBox.Draw(window);
            grabModeCheckBox.Draw(window);
            grabRadiusSlider.Draw(window);
            GSlider.Draw(window);
            speedSlider.Draw(window);
            text1.Draw(window);

            double ke = ComputeKineticEnergy(gravityObjects);
            double pe = ComputePotentialEnergy(gravityObjects, tree, 0.5f);
            double total = ke + pe;

            energyStream.str("");
            energyStream << std::fixed << std::setprecision(3)
                         << "KE:    " << ke    << "\n"
                         << "PE:    " << pe    << "\n"
                         << "Total: " << total << "\n"
                         << "N:     " << gravityObjects.size() << "\n";
            energyText.SetString(energyStream.str());
            energyText.Draw(window);
        }

        if (!doGrabCarefully) {
            grabRadiusCircle.setRadius(worldToScreenLength(grabRadius, screenWidth, 20.f));
            float radius = grabRadiusCircle.getRadius();
            grabRadiusCircle.setOrigin(sf::Vector2f(radius, radius));
            grabRadiusCircle.setPosition(mousePosition);
            window.draw(grabRadiusCircle);
        }

        window.display();
    }
    return 0;
}
