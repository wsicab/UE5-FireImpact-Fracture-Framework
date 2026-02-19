// FractureActor01.cpp

#include "FractureActor01.h"


AFractureActor01::AFractureActor01()
{
    PrimaryActorTick.bCanEverTick = true;

    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    FractureMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("FractureMesh"));
    FractureMesh->SetupAttachment(RootComponent);
    FireComponent = CreateDefaultSubobject<UFireComponent>(TEXT("FireComponent"));

    Index_A = 0;
    Index_B = 1;
    Index_C = 2;
    IgnitionTemperatureMultiplier = 2.0f;
}



void AFractureActor01::BeginPlay()
{
    Super::BeginPlay();

    // 1. mesh generate
    if (Vertices.Num() > 0 && Triangles.Num() > 0)
    {
        GenerateMesh();
        UE_LOG(LogTemp, Warning, TEXT("Mesh generate"));
    }

    // 2. tcp (blueprint)

    // 3. FireComponent 
    if (!FireComponent || !FractureMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("FireComponent or FractureMesh null!"));
        return;

    }
    FireComponent->TargetMeshComponent = FractureMesh;


    // delegate binding
    FireComponent->OnVertexBurnedOut.Clear();
    FireComponent->OnVertexBurnedOut.AddDynamic(this, &AFractureActor01::OnVertexBurnedOut);

    UE_LOG(LogTemp, Warning, TEXT("Delegate binding success"));

    FireComponent->InitializeFireSystem();



    // start fire
    GetWorldTimerManager().SetTimer(InitFireTimerHandle, [this]()
        {
            // actor check
            if (!IsValid(this))
            {
                UE_LOG(LogTemp, Warning, TEXT("Actor destroyed before timer executed"));
                return;
            }

            if (FireComponent && Vertices.Num() > 0)
            {
                UE_LOG(LogTemp, Error, TEXT("start at %d "), Vertices.Num());

                for (int32 i = 0; i < Vertices.Num(); i++)
                {
                    FireComponent->SetIgnitionPoint(i, FireComponent->BurningTemperature * 2.0f);
                }
            }
        }, 2.0f, false);
}


void AFractureActor01::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (IsValid(FireComponent))
    {
        //remove delegate 
        FireComponent->OnVertexBurnedOut.RemoveDynamic(this, &AFractureActor01::OnVertexBurnedOut);

        UE_LOG(LogTemp, Log, TEXT("OnVertexBurnedOut delegate remove"));
    }

    if (UWorld* World = GetWorld())
    {
        if (InitFireTimerHandle.IsValid())
        {
            World->GetTimerManager().ClearTimer(InitFireTimerHandle);
            UE_LOG(LogTemp, Log, TEXT("InitFireTimer"));
        }

        if (ReFireTimerHandle.IsValid())
        {
            World->GetTimerManager().ClearTimer(ReFireTimerHandle);
            UE_LOG(LogTemp, Log, TEXT("ReFireTimer"));
        }

        if (PhysicsTimerHandle.IsValid())
        {
            World->GetTimerManager().ClearTimer(PhysicsTimerHandle);
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("FractureActor01: EndPlay"));

    Super::EndPlay(EndPlayReason);
}



void AFractureActor01::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void AFractureActor01::GenerateMesh()
{
    FractureMesh->CreateMeshSection_LinearColor(0, Vertices, Triangles, Normals, UV0, VertexColors, Tangents, true);

    NewVertices = Vertices;

    if (Recursions > 0)
    {
        for (int i = 0; i < Recursions; ++i)
        {
            for (int j = 0; j < Triangles.Num() / 3; ++j)
            {
                Subdivide(Triangles[Index_A], Triangles[Index_B], Triangles[Index_C]);
            }
            Vertices.Empty();
            Vertices = NewVertices;

            Triangles.Empty();
            Triangles = NewTriangles;
            NewTriangles.Empty();

            Index_A = 0;
            Index_B = 1;
            Index_C = 2;

            VertexDictionary.Empty();
            IndicesDictionary.Empty();
        }
        FractureMesh->CreateMeshSection_LinearColor(0, Vertices, Triangles, Normals, UV0, VertexColors, Tangents, true);
    }
}



void AFractureActor01::Subdivide(int a, int b, int c)
{
    FVector va = Vertices[a];
    FVector vb = Vertices[b];
    FVector vc = Vertices[c];

    FVector vab = FMath::Lerp(va, vb, 0.5);
    FVector vbc = FMath::Lerp(vb, vc, 0.5);
    FVector vca = FMath::Lerp(vc, va, 0.5);

    i_a = a;
    i_b = b;
    i_c = c;

    bool vab_duplicates = false;
    bool vbc_duplicates = false;
    bool vca_duplicates = false;

    for (int i = 0; i < VertexDictionary.Num(); ++i)
    {
        if (vab == VertexDictionary[i])
        {
            vab_duplicates = true;
            i_ab = IndicesDictionary[i];
        }
        if (vbc == VertexDictionary[i])
        {
            vbc_duplicates = true;
            i_bc = IndicesDictionary[i];
        }
        if (vca == VertexDictionary[i])
        {
            vca_duplicates = true;
            i_ca = IndicesDictionary[i];
        }
    }

    if (!vab_duplicates)
    {
        NewVertices.Add(vab);
        VertexDictionary.Add(vab);
        IndicesDictionary.Add(NewVertices.Num() - 1);
        i_ab = NewVertices.Num() - 1;
    }

    if (!vbc_duplicates)
    {
        NewVertices.Add(vbc);
        VertexDictionary.Add(vbc);
        IndicesDictionary.Add(NewVertices.Num() - 1);
        i_bc = NewVertices.Num() - 1;
    }

    if (!vca_duplicates)
    {
        NewVertices.Add(vca);
        VertexDictionary.Add(vca);
        IndicesDictionary.Add(NewVertices.Num() - 1);
        i_ca = NewVertices.Num() - 1;
    }

    NewTriangles.Add(i_a);
    NewTriangles.Add(i_ab);
    NewTriangles.Add(i_ca);

    NewTriangles.Add(i_ca);
    NewTriangles.Add(i_bc);
    NewTriangles.Add(i_c);

    NewTriangles.Add(i_ab);
    NewTriangles.Add(i_b);
    NewTriangles.Add(i_bc);

    NewTriangles.Add(i_ab);
    NewTriangles.Add(i_bc);
    NewTriangles.Add(i_ca);

    Index_A += 3;
    Index_B += 3;
    Index_C += 3;

}


// fire system
void AFractureActor01::OnVertexBurnedOut(int32 VertexIndex, FVector WorldLocation)
{
    if (bFractureInProgress)
    {
        return;
    }

    bFractureInProgress = true;

    if (FireComponent)
    {
        FireComponent->PauseFireSimulation(true);
    }

    UE_LOG(LogTemp, Error, TEXT("BurnedOut fracture verices: %d"), VertexIndex);
    UE_LOG(LogTemp, Warning, TEXT("   world : %s"), *WorldLocation.ToString());


    OnFireFractureTriggered.Broadcast(WorldLocation);

}


void AFractureActor01::UpdateFireComponentMesh()
{
    if (!FireComponent || !FractureMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("UpdateFireComponentMesh: null!"));
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("UpdateFireComponentMesh start"));


    // 1. save fire state
    if (bPreserveFireState)
    {
        SavedFireLocations = FireComponent->SaveFireParticleLocations();
        UE_LOG(LogTemp, Warning, TEXT(" %d fire state saving"), SavedFireLocations.Num());
    }

    // 2. reset temperature (particle)

    FireComponent->ResetTemperaturesOnly();

    // 3. mesh reference update
    FireComponent->TargetMeshComponent = FractureMesh;
    FireComponent->ForceReinitialize();



    // 4. refire (delay)
    GetWorldTimerManager().SetTimer(ReFireTimerHandle, [this]()
        {
            // actor check
            if (!IsValid(this))
            {
                UE_LOG(LogTemp, Warning, TEXT(" Actor destroyed before ReFireTimer executed"));
                return;
            }

            if (FireComponent)
            {
                FProcMeshSection* Section = FractureMesh->GetProcMeshSection(0);
                if (Section)
                {
                    int32 NumVertices = Section->ProcVertexBuffer.Num();

                    if (bPreserveFireState && SavedFireLocations.Num() > 0)
                    {
                      
                        FireComponent->RestoreFireAtLocations(
                            SavedFireLocations,
                            FireComponent->BurningTemperature * 1.2f
                        );

                        FireComponent->PauseFireSimulation(false);
                    }
                    else
                    {
                       // start at center
                        int32 CenterIndex = NumVertices / 2;
                        int32 IgnitionRange = FMath::Min(50, NumVertices / 10);

                        for (int32 i = CenterIndex - IgnitionRange; i <= CenterIndex + IgnitionRange; i++)
                        {
                            if (i >= 0 && i < NumVertices)
                            {
                                FireComponent->SetIgnitionPoint(i, FireComponent->BurningTemperature * 1.2f);
                            }
                        }
                        UE_LOG(LogTemp, Error, TEXT("%d vertices igniting complete"), IgnitionRange * 2 + 1);
                    }
                }
            }
        }, 0.5f, false);
}

void AFractureActor01::IgniteAtIndices(const TArray<int32>& IndicesToIgnite)
{
    if (!FireComponent || Vertices.Num() == 0)
    {
        int32 CurrentVertexCount = 0;
        if (FractureMesh && FractureMesh->GetProcMeshSection(0))
        {
            CurrentVertexCount = FractureMesh->GetProcMeshSection(0)->ProcVertexBuffer.Num();
        }
        UE_LOG(LogTemp, Error, TEXT("IgniteAtIndices: None FireComponent"), CurrentVertexCount);
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("try to ignite %d vertices"), IndicesToIgnite.Num());

    // vertics number
    int32 NumVertices = FractureMesh->GetProcMeshSection(0)->ProcVertexBuffer.Num();

    //  ignite num 0 (if not activate)
    if (IndicesToIgnite.Num() == 0)
    {
        if (NumVertices > 0)
        {
            FireComponent->SetIgnitionPoint(0, FireComponent->BurningTemperature * IgnitionTemperatureMultiplier);
        }
        return;
    }

    // ignite to index
    for (int32 VertexIndex : IndicesToIgnite)
    {
        if (VertexIndex >= 0 && VertexIndex < NumVertices)
        {
            FireComponent->SetIgnitionPoint(VertexIndex, FireComponent->BurningTemperature * IgnitionTemperatureMultiplier);
        }
    }
}



void AFractureActor01::BP_OnFractureComplete(int32 NumPieces)
{

    bFractureInProgress = false;

    if (bEnablePhysicsOnFracture && FractureMesh)
    {
        GetWorldTimerManager().SetTimer(PhysicsTimerHandle, [this]()
            {
                // check actor valid
                if (!IsValid(this))
                {
                    UE_LOG(LogTemp, Warning, TEXT("Actor destroyed before PhysicsTimer executed"));
                    return;
                }

                if (FractureMesh)
                {
                    FractureMesh->SetSimulatePhysics(true);
                    FractureMesh->SetEnableGravity(true);
                    FractureMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                    
                }
            }, PhysicsActivationDelay, false);
    }

    OnFractureDataReceived.Broadcast(NumPieces);

}