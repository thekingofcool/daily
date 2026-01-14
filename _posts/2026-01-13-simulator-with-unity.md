---
layout: post
title: 使用 Unity 构建模拟器
date: 2026-01-13
author: thekingofcool
description: ""
categories: thoughts
---

### Overview
一个模拟器世界观由三部分构成：物体（Object），物理规则（Physical Limit），逻辑脚本（Script）。

- 物体通过建模导入；

- 物理规则是整个世界需要满足的刚性限制，比如重力，这部分通常游戏引擎会提供相应组件；

- 玩家行为和物体反应之间的交互逻辑使用脚本体现，在 Unity 中使用 C# 语言进行描述。

Unity 中逻辑脚本主要分为两个部分：Start() 包含程序启动时会加载的逻辑；Update() 中放置程序运行过程中不断作用的交互逻辑。

### Get Started
首先在 Unity Editor 中创建两个 3D Object - Plane & Sphere；将 Plane 的坐标位置设置在原点 (0, 0, 0)，Sphere 的坐标位置放在 (0, 1, 0)；给 Sphere 添加重力组件 Rigidbody；编写脚本使 Sphere 初始加载重力，随后接收到键入空格信号则向上作用一个冲力。

BallController.cs

```c#
using UnityEngine;
using UnityEngine.InputSystem;
public class BallController : MonoBehaviour  
{  
    public float jumpForce = 5f;  
    private Rigidbody rb;
    void Start()  
    {  
        rb = GetComponent<Rigidbody>();  
    }  
    void Update()  
    {  
    if (Keyboard.current.spaceKey.wasPressedThisFrame)  
    {  
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);  
    } 
    }  
}
```

接着给球体添加 Script 组件，选择上方脚本，运行项目即可测试初始功能。

接着给球添加外观。在项目 Assets 目录中创建 Material，在 Surface Inputs 中修改 Base Map 颜色，将这个材料组件添加给球体。

然后设置规则：只有当球在地面上才能跳起。对此需要增加变量记录球体与地面接触的状态。定义球体和平台碰撞（接触）时，接触状态为“是”；当按下空格的瞬间，接触状态变为“否”。

BallController.cs

```c#
using UnityEngine;
using UnityEngine.InputSystem;
public class BallController : MonoBehaviour
{
    public float jumpForce = 5f;
    private Rigidbody rb;
    public bool isGrounded;
    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }
    void Update()
    {
    if (Keyboard.current.spaceKey.wasPressedThisFrame && isGrounded)
    {
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        isGrounded = false;
    }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Plane")
        {
            isGrounded = true;
        }
    }
}
```

按照这个规则，将元素进一步扩充：玩家通过键盘按键改变平面移动方向，空格是跳跃，鼠标移动改变视角，点击鼠标发射子弹，子弹击中物体使物体消失。

为此需要:

1. 把平面扩大；
2. 生成一个角色，这个角色受重力作用，但移动不至于倾倒；
3. 平台上不规则布置一些目标；
4. 因为是第一人称视角，将主相机置于角色元素的一部分；
5. 需要有一套无限生成的子弹，通过点击事件发出，击中物体则消失，未击中物体限时销毁；
6. 子弹击中除目标以外物体不作用。

PlayerController.cs

```c#
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    [Header("移动设置")]
    public float moveSpeed = 7f;
    public float jumpForce = 5f;

    [Header("视角设置")]
    public float mouseSensitivity = 15f;
    public Transform playerCameraTransform;
    
    [Header("射击设置")]
    public GameObject bulletPrefab;
    public Transform bulletSpawnPoint;
    public float bulletSpeed = 40f;

    private Rigidbody rb;
    private float xRotation = 0f;
    private bool isGrounded;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // 1. 隐藏鼠标并锁定到屏幕中心
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        // 2. 初始视角重置：强制平视前方
        if (playerCameraTransform != null)
        {
            playerCameraTransform.localRotation = Quaternion.Euler(0, 0, 0);
        }
        xRotation = 0f;
    }

    void Update()
    {
        // 每一帧处理：视角旋转、跳跃检测、射击
        HandleLook();
        HandleJump();
        HandleShoot();
    }

    void FixedUpdate()
    {
        // 每物理帧处理：移动
        HandleMovement();
    }

    void HandleLook()
    {
        // 获取鼠标移动增量
        Vector2 mouseDelta = Mouse.current.delta.ReadValue();
        
        // 左右看：旋转角色的 Y 轴（身体转）
        float mouseX = mouseDelta.x * mouseSensitivity * Time.deltaTime;
        transform.Rotate(Vector3.up * mouseX);

        // 上下看：旋转相机的 X 轴（头转）
        float mouseY = mouseDelta.y * mouseSensitivity * Time.deltaTime;
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -80f, 80f); // 限制仰角和俯角

        playerCameraTransform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
    }

    void HandleMovement()
    {
        Vector2 input = Vector2.zero;
        var kb = Keyboard.current;

        if (kb.wKey.isPressed) input.y += 1;
        if (kb.sKey.isPressed) input.y -= 1;
        if (kb.aKey.isPressed) input.x -= 1;
        if (kb.dKey.isPressed) input.x += 1;

        if (input.magnitude > 1) input.Normalize();

        // 核心：基于角色当前的“面朝方向”来计算移动向量
        // transform.forward 是角色当前正对着的方向
        Vector3 moveDir = (transform.forward * input.y + transform.right * input.x);
        
        // 保留原有的垂直速度（重力），只修改水平移动
        Vector3 newVelocity = moveDir * moveSpeed;
        newVelocity.y = rb.linearVelocity.y; // Unity 6 中 Rigidbody 使用 linearVelocity
        
        rb.linearVelocity = newVelocity;
    }

    void HandleJump()
    {
        if (Keyboard.current.spaceKey.wasPressedThisFrame && isGrounded)
        {
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
            isGrounded = false;
        }
    }

    void HandleShoot()
    {
        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            if (bulletPrefab != null && bulletSpawnPoint != null)
            {
                GameObject bullet = Instantiate(bulletPrefab, bulletSpawnPoint.position, bulletSpawnPoint.rotation);
                Rigidbody bulletRb = bullet.GetComponent<Rigidbody>();
                // 子弹朝相机正前方飞出
                bulletRb.linearVelocity = playerCameraTransform.forward * bulletSpeed;
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Ground"))
        {
            isGrounded = true;
        }
    }
}
```

Bullet.cs

```c#
using UnityEngine;

public class Bullet : MonoBehaviour
{
    public float lifetime = 3f;
    void Start()
    {
        // 3秒后自动销毁子弹
        Destroy(gameObject, lifetime);
    }
    private void OnCollisionEnter(Collision collision)
    {
        // 检查撞到的物体是不是带了 "Target" 标签
        if (collision.gameObject.CompareTag("Target"))
        {
            Destroy(collision.gameObject);
        }
        Destroy(gameObject);
    }
}
```

### Capability
Unity 引擎的 Package Manager 是为开发者实现降本增效的关键，它大体分为三类：

1. 渲染管线架构: Universal Render Pipeline(URP), High Definition Render Pipeline(HDRP)。只需维护一套逻辑，就能同时占领低算力和高算力市场，相比于 Unreal 更注重高端，这是 Unity 一个显著的差异化优势。

2. 面向数据的技术栈(DOTS): Entities, Burst Compiler, C# Job System。对底层重构，允许代码默认利用多核处理器，让普通设备也能运行以前只有超高端设备才能运行的复杂场景，同屏渲染海量单元而不卡顿，这对于大型开放世界模拟和工业数字孪生至关重要。

3. 游戏开发，运营，变现形成闭环：Unity Ads, LevelPlay, Unity Gaming Services(UGS)。改变公司收入结构，通过提供多人联网、云存档、分析以及广告聚合等功能，从单纯引擎订阅过渡为引擎订阅 & 广告盈利 & 高价值用户获取 & 云服务的全生命周期介入的盈利模式。

另外，Unity Asset Store 里丰富的成熟模块、庞大的开发者生态社群以及引擎对 AI 的整合（开发辅助 & 端侧运行 NPC AI 模型）对提升开发效率同样不可忽视。这家公司的愿景是推动技术平权，让小团队开发者拥有不亚于大型工作室的产出能力。

### Thoughts
马斯克说这个世界大概率是一个模拟器，我们能做的就是有点幽默感，因为如果太无聊了高维生物会把服务器关掉。你无法说这是一个笑话，因为就目前为止没人能够验证我们是否处在模拟器中。

一个也许可行的验证方式就是去别的星球或者星系再造一个文明，增大服务器的荷载如果导致物理世界出现“故障”那就坐实这一点。

另外一个验证方法就是如果人类能够创造某种逼真的宇宙，为其中定义规则（物理规律），让物质依照规则相互作用，让文明不断演化，由此获得某种方面的乐趣。如果这个被证明可行，那也从另一个角度证明了这一点。

参考资料：

[Rick and Morty S1E4](https://www.imdb.com/title/tt3333830/){:target="_blank"}

[Rick and Morty S2E6](https://www.imdb.com/title/tt4832268/){:target="_blank"}
