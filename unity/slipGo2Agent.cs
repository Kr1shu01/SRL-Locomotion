using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System;
using System.Collections.Specialized;

public class slipGo2Agent : Agent
{
    float[] uff = new float[12] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] contact = new float[4] { 0, 0, 0, 0 };
    float[] u = new float[12] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] u0 = new float[12] { 0, 30f, -62f, 0, 30f, -62f, 0, 30f , -62f , 0, 30f , -62f };
    float[] utotal = new float[12] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[,] angs = new float[12, 200];
    float[,] angr = new float[12, 500];

    public PhysicMaterial groundMaterial;
    public Collider groundCollider;
    public float randomStiffness = 0f;
    public float randomDamping = 0f;


    private string filePath = @"D:\Unity\packages\ml-agents-release_21\Project\Assets\ML-Agents\Examples\Articulation\SLIP\data\uff12dof.csv";
    //private string filePath = @"D:\Unity\packages\ml-agents-release_21\Project\Assets\ML-Agents\Examples\Articulation\SLIP\data\randomquad.csv";
    private List<string> dataBuffer = new List<string>();
    //private List<string> dataBuffer1 = new List<string>();

    [Header("Body Parts")]
    public Transform trunk;
    public Transform rl_hip;
    public Transform rl_thigh;
    public Transform rl_calf;
    public Transform rl_foot;
    public Transform rr_hip;
    public Transform rr_thigh;
    public Transform rr_calf;
    public Transform rr_foot;
    public Transform fl_hip;
    public Transform fl_thigh;
    public Transform fl_calf;
    public Transform fl_foot;
    public Transform fr_hip;
    public Transform fr_thigh;
    public Transform fr_calf;
    public Transform fr_foot;
    public Transform ft1;

    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Vector3 temp;
    Quaternion rot0;
    ArticulationBody[] jh = new ArticulationBody[12];
    ArticulationBody[] foot = new ArticulationBody[4];
    ArticulationBody art0;
    List<float> positions = new List<float>();
    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();

    int tp = 0;
    int tq = 0;
    int n = 0;
    float dr = 0;
    int state = 0;
    Vector3 pf = Vector3.zero;
    Vector3 pm = Vector3.zero;
    Vector3 vm = Vector3.zero;
    Vector3 Fn0 = Vector3.zero;

    [Header("SLIP")]
    public Transform mass;
    public Transform ft;
    Vector3 pm0;
    Vector3 pf0;
    Rigidbody rbms;

    public override void Initialize()
    {
        bodypart.Add(rl_hip);
        bodypart.Add(rl_thigh);
        bodypart.Add(rl_calf);
        bodypart.Add(rr_hip);
        bodypart.Add(rr_thigh);
        bodypart.Add(rr_calf);

        bodypart.Add(fl_hip);
        bodypart.Add(fl_thigh);
        bodypart.Add(fl_calf);
        bodypart.Add(fr_hip);
        bodypart.Add(fr_thigh);
        bodypart.Add(fr_calf);

        art0 = trunk.GetComponent<ArticulationBody>();
        var i = 0;
        foreach (var bp in bodypart)
        {
            jh[i] = bp.GetComponent<ArticulationBody>();
            i++;
        }
        pos0 = trunk.position;
        rot0 = trunk.rotation;
        art0.GetJointPositions(positions);
        art0.GetJointPositions(P0);
        art0.GetJointVelocities(W0);

        rbms = mass.GetComponent<Rigidbody>();
        pm0 = mass.position;
        pf0 = ft.position;
    }

    public override void OnEpisodeBegin()
    {
        art0.SetJointPositions(P0);
        art0.SetJointVelocities(W0);
        art0.TeleportRoot(pos0, rot0);
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
        for (int i = 0; i < 12; i++) SetJointTargetDeg(jh[i], u0[i]);

        for (int i = 0; i < 12; i++) u[i] = 0;
        for (int i = 0; i < 12; i++) uff[i] = 0;


        tp = 0;
        tq = 0;
        state = -1;
        mass.position = pm0;
        rbms.velocity = Vector3.zero;
        ft.position = pf0;
        n++;

        //RandomizeFriction();
        if (!File.Exists(filePath))
        {
            File.WriteAllText(filePath, "\n");
            //File.WriteAllText(filePath1, "\n");
        }
    }

    public override void CollectObservations(VectorSensor sensor)//32-3维
    {
        var vel = trunk.InverseTransformDirection(art0.velocity);
        var wel = trunk.InverseTransformDirection(art0.angularVelocity);
        float noiseStdDev = Random.Range(0.001f, 0.020f);
        //float noiseStdDev = Random.Range(0f, 0f);

        sensor.AddObservation(EulerTrans(trunk.eulerAngles[0]) * 3.14f / 180f + Random.Range(- noiseStdDev, noiseStdDev));//rad
        sensor.AddObservation(EulerTrans(trunk.eulerAngles[2]) * 3.14f / 180f + Random.Range(- noiseStdDev, noiseStdDev));//rad
        sensor.AddObservation(vel - vm);
        sensor.AddObservation(wel);
        for (int i = 0; i < 12; i++)
        {
            //sensor.AddObservation((jh[i].jointPosition[0]) - uff[i] * 3.14f / 180f + Random.Range(- noiseStdDev, noiseStdDev));
            sensor.AddObservation((jh[i].jointPosition[0]) - uff[i] * 3.14f / 180f + Random.Range(-noiseStdDev, noiseStdDev));
            sensor.AddObservation(jh[i].jointVelocity[0] + Random.Range(-noiseStdDev, noiseStdDev));
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
        var haa = 10f;
        var hfe = 30f;
        var kfe = 60f;
        float[] kb = new float[12] { haa, hfe, kfe, haa, hfe, kfe, haa, hfe, kfe, haa, hfe, kfe };
        for (int i = 0; i < 12; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            utotal[i] = kb[i] * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }
    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        //drive.forceLimit = 100f;
        drive.target = x;
        joint.xDrive = drive;
    }
    float Tanh(float x)
    {
        return (Mathf.Exp(x) - Mathf.Exp(-x)) / (Mathf.Exp(x) + Mathf.Exp(-x));
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
    }
    //////////////////////////////////////////////////////////////////////////////
    void FixedUpdate()
    {
        tp++;
        tq++;
        pf = ft.position;//橙色目标足端位置[位于左前足端][小球]
        pm = mass.position;//橙色目标Mass位置[位于左前大腿][大球]
        //pm = fl_thigh.position;
        vm = rbms.velocity;//橙大球速度
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
                //dr = 0.005f * Random.Range(0, 4);
                dr = 0.01f;
                pf.z = pf.z - dr;
                ft.position = pf;
                n++;
            }
        }
        if (state == 0)
        {
            F = -90 * dl * fn;
            //F = -60 * dl * fn;
            if (dl < -0.05f && vm.y > 0) state = 1;
        }
        if (state == 1)
        {
            //F = -200 * (dl - 0.1f) * fn;
            //F = -200 * dl * fn;
            F = -300 * dl * fn;//此处调整跳跃高度
            if (dl >= -0.05f)
            {
                state = 2;
                Fn0 = pf - pm;
            }
        }
        if (state == 2)
        {
            //var dp = pf0 - pm0;
            //dp.y += 0.35f;
            F = Vector3.zero;
            //ft.position = pm + dp;// 0.5f * pf + 0.5f * (pm + dp);
            ft.position = pm + Fn0;
            //Fn0.z = 0.8f * Fn0.z + 0.02f * vm.z;//跳跃高度修改后要调整参数以使稳定
            Fn0.z = 0.75f * Fn0.z + 0.02f * vm.z;
            Fn0.x = 0;
            Fn0.y = -Mathf.Sqrt(L * L - Fn0.z * Fn0.z);
            if (vm.y < 0 && pf.y <= 0.025f)
            {
                pf.y = 0.025f;
                ft.position = pf;
                state = 3;
            }
        }
        if (state == 3)
        {
            F = -400 * dl * fn;//下落时的缓冲力
            if (vm.y > 0) state = 4;
        }
        if (state == 4)
        {
            F = -100 * ((pm - pf) - (pm0 - pf0)) - 20 * vm;
            F.y += 9.8f;
            if ((pm - pf).magnitude - L > -0.02f)
            {
                state = -1;
            }
        }
        //print(state);
        F.x = -100 * (pm - pm0).x - 20 * vm.x;
        //F.z = -100 * (pm - pm0).z - 20 * vm.z;
        rbms.AddForce(F, ForceMode.Force);
        ft1.position = fl_thigh.position + pf - pm;//ft1为绿色足端应达位置[根据绿色大球即fl_thigh左前大腿位置确定]
        ///////////////////////////////////////////////////////////////////////////////////////////
        var pr = ft1.position - fl_thigh.position;
        //var d1 = pr.magnitude;
        //var L1 = 0.2132f;
        //var L2 = 0.2164f;
        var a1 = Mathf.Acos(Mathf.Clamp(pr.magnitude / 0.426f, -1f, 1f));
        //var a1 = Mathf.Acos(Mathf.Clamp((L1 * L1 + dl * dl - L2 * L2) / (2 * L1 * d1), -1f, 1f));
        var a2 = 3.14f - 2 * a1;
        var da1 = -Mathf.Atan(-pr.z / pr.y) + a1;
        var da2 = -2 * a1;
        uff[0] = 0;
        uff[1] = da1 * 180f / 3.14f;
        uff[2] = da2 * 180f / 3.14f;
        uff[3] = 0;
        uff[4] = da1 * 180f / 3.14f;
        uff[5] = da2 * 180f / 3.14f;
        uff[6] = 0;
        uff[7] = da1 * 180f / 3.14f;
        uff[8] = da2 * 180f / 3.14f;
        uff[9] = 0;
        uff[10] = da1 * 180f / 3.14f;
        uff[11] = da2 * 180f / 3.14f;
        
        //////////////////////////////////////////////////////////////////////////////////////
        var vel = trunk.InverseTransformDirection(art0.velocity);
        var wel = trunk.InverseTransformDirection(art0.angularVelocity);
        var live_reward = 1f;
        var ori_reward1 = -0.02f * Mathf.Min(Mathf.Abs(trunk.eulerAngles[0]), Mathf.Abs(trunk.eulerAngles[0] - 360f));
        var ori_reward2 = -2f * Mathf.Abs(wel[1]); // -0.2f * Mathf.Min(Mathf.Abs(body.eulerAngles[1]), Mathf.Abs(body.eulerAngles[1] - 360f));//-2f*Mathf.Abs(art0.angularVelocity[1]);// 
        var ori_reward3 = -0.02f * Mathf.Min(Mathf.Abs(trunk.eulerAngles[2]), Mathf.Abs(trunk.eulerAngles[2] - 360f));
        var ori_reward = ori_reward1 + ori_reward2 + ori_reward3;

        //var vel_reward = -Mathf.Abs(vel[0]) + Mathf.Abs(vel[2]);// - 01f * (vel - vm).magnitude;
        var vel_reward = -1f * (art0.velocity - vm).magnitude;
        var pos_reward1 = -(fl_foot.position - fl_thigh.position - pr).magnitude;
        var pos_reward2 = -(rl_foot.position - rl_thigh.position - pr).magnitude;
        var pos_reward3 = -(fr_foot.position - fr_thigh.position - pr).magnitude;
        var pos_reward4 = -(rr_foot.position - rr_thigh.position - pr).magnitude;
        var track_reward = pos_reward1 + pos_reward2 + pos_reward3 + pos_reward4;
        var syn_reward = -Mathf.Abs(jh[1].jointPosition[0] - jh[4].jointPosition[0]) - Mathf.Abs(jh[2].jointPosition[0] - jh[5].jointPosition[0]) - Mathf.Abs(jh[7].jointPosition[0] - jh[10].jointPosition[0]) - Mathf.Abs(jh[8].jointPosition[0] - jh[11].jointPosition[0]);
        //if (gait == 3) vel_reward = Mathf.Abs(vel[1])- Mathf.Abs(vel[2]);
        //if (gait == 4) vel_reward = -Mathf.Abs(vel[1]) - Mathf.Abs(vel[2]);
        //vel_reward = -5*Mathf.Abs(vel[1]);
        //var u_reward = 0f;
        //for (int i = 0; i < 12; i++) u_reward -= (u[i] * 3.14f / 180f) * (u[i] * 3.14f / 180f);
        var h_reward = Mathf.Clamp((trunk.position - pos0).y - 0.1f, 0, 1);
        var d_reward = (trunk.position - pos0).z;
        var reward = live_reward + 1f * track_reward + 1f * vel_reward + 1f * ori_reward + 0 * h_reward + 0f * d_reward + 1f * syn_reward;
        AddReward(reward);
        /*if ((pos0 - trunk.position).y > 0.35f)
        {
            //EndEpisode();
        }*/
        if (Mathf.Min(Mathf.Abs(trunk.eulerAngles[0]), Mathf.Abs(trunk.eulerAngles[0] - 360f)) > 20f)
        {
            EndEpisode();
        }
        if (Mathf.Min(Mathf.Abs(trunk.eulerAngles[2]), Mathf.Abs(trunk.eulerAngles[2] - 360f)) > 20f)
        {
            EndEpisode();
        }

        if (Time.fixedTime <= 10f)
        {
            string data = Time.fixedTime.ToString("F2") + ",";  // 时间戳
            data = da1 + "," + da2 ;
            dataBuffer.Add(data);
        }
        else
        {
            if (dataBuffer.Count > 0)
            {
                WriteDataToFile();
}
        }

        /*if (Time.fixedTime <= 10f)
        {
            string data = Time.fixedTime.ToString("F2") + ",";  // 时间戳
            /*string data1 = Time.fixedTime.ToString("F2") + ",";  // 时间戳

            float trunkPositionError = 0f;
            float FLFootEndError = 0f; // 左前足端误差 (Front Left)
            float FRFootEndError = 0f; // 右前足端误差 (Front Right)
            float RLFootEndError = 0f; // 左后足端误差 (Rear Left)
            float RRFootEndError = 0f; // 右后足端误差 (Rear Right)

            var pr = ft1.position - fl_thigh.position;
            trunkPositionError = Vector3.Distance(fl_thigh.position, mass.position);
            FLFootEndError = (fl_foot.position - fl_thigh.position - pr).magnitude;
            FRFootEndError = (fr_foot.position - fr_thigh.position - pr).magnitude;
            RLFootEndError = (rl_foot.position - rl_thigh.position - pr).magnitude;
            RRFootEndError = (rr_foot.position - rr_thigh.position - pr).magnitude;
            data += trunkPositionError + "," + FLFootEndError + "," + FRFootEndError + "," + RLFootEndError + "," + RRFootEndError + ",";
            data = uff[1] + "," + uff[2];
            //var vel = trunk.InverseTransformDirection(art0.velocity);
            //data1 += vel.magnitude + "," + vm.magnitude + ",";

            //data1 += body.position.y + "," + foot_l.position.y + "," + foot_r.position.y + ",";

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
    private void RandomizeFriction()
    {
        groundMaterial.dynamicFriction = Random.Range(0.5f, 1f);
        groundMaterial.staticFriction = Random.Range(0.5f, 1f);
        groundMaterial.bounciness = Random.Range(0f, 0.1f);
        groundCollider.material = groundMaterial;
    }
}
