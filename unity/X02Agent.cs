using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

public class X02Agent : Agent
{
    int tp = 0;
    int tq = 0;
    //int T1 = 30;
    int stop = 0;
    int state = -1;
    float dr = 0;
    int n = 0;
    Vector3 Fn0 = Vector3.zero;
    Vector3 pf = Vector3.zero;
    Vector3 pm = Vector3.zero;
    Vector3 vm = Vector3.zero;
    float[] uff = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] u = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] u0 = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] utotal = new float[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float kb = 1f;
    float kf = 1f;
    float[] DD = new float[24] { 0.03f, 0.01f, 0.02f, 0.02f, 0.03f, 0.01f, 0.03f, 0.02f, 0.01f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    private string filePath = @"D:\Unity\packages\ml-agents-release_21\Project\Assets\ML-Agents\Examples\Articulation\SLIP\data\pxpz0213.csv";
    private List<string> dataBuffer = new List<string>();
    //private List<string> dataBuffer1 = new List<string>();

    [Header("Body Parts")]
    public Transform body;
    public Transform yaw_l;
    public Transform hip_l;
    public Transform thigh_l;
    public Transform shank_l;
    public Transform foot_l;
    public Transform yaw_r;
    public Transform hip_r;
    public Transform thigh_r;
    public Transform shank_r;
    public Transform foot_r;
    public Transform mass;
    public Transform ft;
    public Transform mass1;
    public Transform ft1;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Vector3 posfl0;
    Vector3 posfr0;
    Vector3 pm0;
    Vector3 pf0;
    Quaternion rot0;
    ArticulationBody[] jh = new ArticulationBody[10];
    ArticulationBody art0;
    Rigidbody rbms;

    public override void Initialize()
    {
        bodypart.Add(yaw_l);//yaw
        bodypart.Add(hip_l);//haal
        bodypart.Add(thigh_l);//hfel
        bodypart.Add(shank_l);//kneel
        bodypart.Add(foot_l);//ankl
        bodypart.Add(yaw_r);//yaw
        bodypart.Add(hip_r);//haar
        bodypart.Add(thigh_r);//hfer
        bodypart.Add(shank_r);//kneer
        bodypart.Add(foot_r);//ankr

        art0 = body.GetComponent<ArticulationBody>();
        rbms = mass.GetComponent<Rigidbody>();
        var i = 0;
        foreach (var bp in bodypart)
        {
            jh[i] = bp.GetComponent<ArticulationBody>();
            u0[i] = jh[i].jointPosition[0] * 180f / 3.14f;
            uff[i] = u0[i];
            i++;
        }
        pos0 = body.position;
        rot0 = body.rotation;
        art0.GetJointPositions(P0);
        art0.GetJointVelocities(W0);

        pm0 = mass.position;
        pf0 = ft.position;
        posfl0 = foot_l.position;
        posfr0 = foot_r.position;
    }

    public override void OnEpisodeBegin()
    {
        tp = 0;
        tq = 0;
        stop = Random.Range(0, 3);
        art0.TeleportRoot(pos0, rot0);
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
        art0.SetJointPositions(P0);
        art0.SetJointVelocities(W0);
        for (int i = 0; i < 10; i++) u[i] = 0;

        state = -1;
        mass.position = pm0;
        rbms.velocity = Vector3.zero;
        ft.position = pf0;
        pf = ft.position;
        pm = mass.position;
        vm = rbms.velocity;
        //dr = 0.01f * Random.Range(0, 4);
        //dr = 0.03f;

        /*if (!File.Exists(filePath))
        {
            File.WriteAllText(filePath, "\n");
            //File.WriteAllText(filePath1, "\n");
        }*/
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var vel = body.InverseTransformDirection(art0.velocity);
        var wel = body.InverseTransformDirection(art0.angularVelocity);
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * 3.14f / 180f );//rad
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f );//rad
        sensor.AddObservation(vel - vm);
        sensor.AddObservation(wel);
        for (int i = 0; i < 10; i++)
        {
            sensor.AddObservation((jh[i].jointPosition[0]) - uff[i] * 3.14f / 180f);
            sensor.AddObservation(jh[i].jointVelocity[0]);
        }
    }
    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180)
            return eulerAngle;
        else
            return eulerAngle - 360f;
    }
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        kf = 1f;
        kb = 30f;
        for (int i = 0; i < 10; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            utotal[i] = kb * u[i] + kf * uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }
    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 1000f;//400f;//180f
        drive.damping = 50f;//20;// 8f;
        //drive.forceLimit = 100f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
    }

    void FixedUpdate()
    {
        tp++;
        tq++;
        var vel = body.InverseTransformDirection(art0.velocity);
        var wel = body.InverseTransformDirection(art0.angularVelocity);
        var posb = mass1.position;//posb[mass1]为绿色本体body位置，pm[mass]是橙色目标body位置
        var posfl = foot_l.position;//此为实际足端位置，ft1为绿色应达足端位置
        var posfr = foot_r.position;
        var pfl = pf;//pf为橙色目标足端位置
        var pfr = pf;
        pfl.x = posfl0.x;
        pfr.x = posfr0.x;

        var live_reward = 1f;
        var ori_reward1 = -0.1f * Mathf.Min(Mathf.Abs(body.eulerAngles[0]), Mathf.Abs(body.eulerAngles[0] - 360f));
        var ori_reward2 = -0.5f * Mathf.Abs(wel[1]); // -0.2f * Mathf.Min(Mathf.Abs(body.eulerAngles[1]), Mathf.Abs(body.eulerAngles[1] - 360f));//-2f*Mathf.Abs(art0.angularVelocity[1]);// 
        var ori_reward3 = -0.1f * Mathf.Min(Mathf.Abs(body.eulerAngles[2]), Mathf.Abs(body.eulerAngles[2] - 360f));
        var ori_reward = ori_reward1 + ori_reward2 + ori_reward3;

        var syn_reward = -1f * Mathf.Abs(jh[2].jointPosition[0] - jh[7].jointPosition[0]) - 1f * Mathf.Abs(jh[3].jointPosition[0] - jh[8].jointPosition[0]) - 1f * Mathf.Abs(jh[4].jointPosition[0] - jh[9].jointPosition[0]);
        var pos_reward = -1f * ((posb - posfl - pm + pfl).magnitude + (posb - posfr - pm + pfr).magnitude);
        var vel_reward = -1f * (art0.velocity - vm).magnitude;

        //var reward = live_reward + 1f * pos_reward + 1f * vel_reward + 1f * ori_reward + 1f * syn_reward;
        var reward = live_reward + 1f * pos_reward + 0.5f * vel_reward + 2f * ori_reward + 0f * syn_reward;
        //var reward = live_reward + 1f * pos_reward + 1f * vel_reward + 1f * ori_reward + 0f * syn_reward;
        AddReward(reward);

        /*if (pos_reward < -1f)
        {
            EndEpisode();
        }*/
        if (Mathf.Min(Mathf.Abs(body.eulerAngles[0]), Mathf.Abs(body.eulerAngles[0] - 360f)) > 30f)
        {
            EndEpisode();
        }
        if (Mathf.Min(Mathf.Abs(body.eulerAngles[2]), Mathf.Abs(body.eulerAngles[2] - 360f)) > 30f)
        {
            EndEpisode();
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        pf = ft.position;
        pm = mass.position;
        vm = rbms.velocity;
        var L = (pm0 - pf0).magnitude;
        var dl = (pm - pf).magnitude - L;
        var fn = (pm - pf) / (pm - pf).magnitude;
        Vector3 F = Vector3.zero;
        if (state == -1)
        {
            F = -100 * ((pm - pf) - (pm0 - pf0)) - 20 * vm;
            F.y += 9.8f;
            if (tq > 20) 
            {
                state = 0;
                //dr = DD[n];
                //dr = 0.02f * Random.Range(0, 4);
                dr = 0.025f;//质心提高0.21m后
                pf.z = pf.z - dr;
                ft.position = pf;
                n++;
            }
        }
        if (state == 0)
        {
            F = -60 * dl * fn;
            if (dl<-0.05f && vm.y > 0) state = 1;
        }
        if (state == 1)
        {
            F = -100f * dl * fn;//-100
            if (dl >= -0.05f)
            {
                state = 2;
                Fn0 = pf - pm;
            }
        }
        if (state == 2)
        {
            F = Vector3.zero;
            ft.position = pm + Fn0;
            Fn0.z = 0.92f * Fn0.z + 0.1f * 0.2f * vm.z;
            Fn0.x = 0;
            Fn0.y = -Mathf.Sqrt(L * L - Fn0.z * Fn0.z);
            if (vm.y < 0 && pf.y <= 0.09f)//0.05f
            {
                pf.y = 0.09f;//0.05f
                ft.position = pf;
                state = 3;
            }
        }
        if (state == 3)
        {
            F = -100f * dl * fn;
            if (vm.y > 0) state = 4;
        }
        if (state == 4)
        {
            F = -100 * ((pm - pf) - (pm0 - pf0)) - 30 * vm;
            F.y += 9.8f;
            if ((pm - pf).magnitude - L > -0.02f)
            {
                state = -1;
            }
        }
        rbms.AddForce(F, ForceMode.Force);
        ft1.position = mass1.position + pf - pm;
        /////////////////////////////////////////////////////////////
        var d1 = (pm-pf).magnitude;
        //var d1 = (mass2.position - ft1.position).magnitude;
        var L1 = 0.4f;//0.4f;
        var L2 = 0.4f;//0.352f;
        var c11 = Mathf.Acos(Mathf.Clamp((L1 * L1 + d1 * d1 - L2 * L2) / (2 * L1 * d1), -1f, 1f));
        var c12 = -Mathf.Asin((pm - pf).z / d1);
        var c31 = Mathf.Acos(Mathf.Clamp((L2 * L2 + d1 * d1 - L1 * L1) / (2 * L2 * d1), -1f, 1f));
        var dc1 = c11 + c12;
        var dc2 = -c11 - c31;
        //var dc3 = dc2 + dc1;
        var dc3 = - dc2 - dc1;

        uff[2] = dc1 * 180f / 3.14f;
        uff[7] = dc1 * 180f / 3.14f; 
        uff[3] = dc2 * 180f / 3.14f; 
        uff[8] = dc2 * 180f / 3.14f;
        uff[4] = dc3 * 180f / 3.14f;
        uff[9] = dc3 * 180f / 3.14f;

        /*uff[2] = 2 * dc1 * 180f / 3.14f / (jh[2].xDrive.upperLimit - jh[2].xDrive.lowerLimit);
        uff[7] = 2 * dc1 * 180f / 3.14f / (jh[7].xDrive.upperLimit - jh[7].xDrive.lowerLimit);
        uff[3] = 0*96f / 126f + 2 * dc2 * 180f / 3.14f / (jh[3].xDrive.upperLimit - jh[3].xDrive.lowerLimit);
        uff[8] = 0*96f / 126f + 2 * dc2 * 180f / 3.14f / (jh[8].xDrive.upperLimit - jh[8].xDrive.lowerLimit);
        uff[4] = 2 * dc3 * 180f / 3.14f / (jh[4].xDrive.upperLimit - jh[4].xDrive.lowerLimit);
        uff[9] = 2 * dc3 * 180f / 3.14f / (jh[9].xDrive.upperLimit - jh[9].xDrive.lowerLimit);*/
        //print(uff[8]);

        /*if (Time.fixedTime <= 10f)
        {
            string data = Time.fixedTime.ToString("F2") + ",";  // 时间戳
            //string data1 = Time.fixedTime.ToString("F2") + ",";  // 时间戳

            /*float trunkPositionError = 0f;
            float leftAnklePositionError = 0f;
            float rightAnklePositionError = 0f;

            var vel = body.InverseTransformDirection(art0.velocity);
            var wel = body.InverseTransformDirection(art0.angularVelocity);
            var posb = mass1.position;//posb[mass1]为绿色本体body位置，pm[mass]是橙色目标body位置
            var posfl = foot_l.position;//此为实际足端位置，ft1为绿色应达足端位置
            var posfr = foot_r.position;
            var pfl = pf;//pf为橙色目标足端位置
            var pfr = pf;
            pfl.x = posfl0.x;
            pfr.x = posfr0.x;

            trunkPositionError = Vector3.Distance(mass1.position, mass.position); // 计算躯干位置误差
            leftAnklePositionError = (posb - posfl - pm + pfl).magnitude; // 计算左脚踝相对位置误差
            rightAnklePositionError = (posb - posfr - pm + pfr).magnitude; // 计算右脚踝相对位置误差

            //data += trunkPositionError + "," + leftAnklePositionError + "," + rightAnklePositionError + ",";
            //data1 += vel.magnitude + "," + vm.magnitude + ",";

            //data1 += body.position.y + "," + foot_l.position.y + "," + foot_r.position.y + ",";  
            data = (pm - pf).z + "," + (pm - pf).y;
            //data = vm.x + "," + vm.y + "," + vm.z;
            //data += "\n";
            //data1 += "\n";
            dataBuffer.Add(data);
            //dataBuffer1.Add(data1);
        }
        else
        {
            if (dataBuffer.Count > 0)
            {
                WriteDataToFile();
            }
        }*/
    }
    float rad_normalize(ArticulationBody joint, float ang)
    {
        var ang1 = ang * 180f / 3.14f;
        var x = (ang1 - joint.xDrive.lowerLimit) / (joint.xDrive.upperLimit - joint.xDrive.lowerLimit);//0~1
        x = 2 * x - 1;//-1~1
        return x;
    }
    float ang_trans(ArticulationBody joint, float x)
    {
        x = (x + 1f) * 0.5f;
        var ang = Mathf.Lerp(joint.xDrive.lowerLimit, joint.xDrive.upperLimit, x);
        return ang;
    }
    private void WriteDataToFile()
    {
        if (dataBuffer.Count > 0)
        {
            File.AppendAllText(filePath, string.Join("\n", dataBuffer) + "\n");
            //File.AppendAllText(filePath, string.Join("\n", dataBuffer1) + "\n");
            dataBuffer.Clear();
            //dataBuffer1.Clear();
        }
    }
}
